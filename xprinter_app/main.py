#!/usr/bin/env python3
"""
XPrinter Order Number Printer
Prints order numbers to a USB receipt printer (XPrinter) on Windows.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
from datetime import datetime

try:
    import win32print
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# ── ESC/POS command constants ──────────────────────────────────────────────────
ESC = b'\x1b'
GS  = b'\x1d'

CMD_INIT          = ESC + b'@'
CMD_CODEPAGE_866  = ESC + b't\x11'   # select CP866 (Cyrillic) on printer
CMD_ALIGN_LEFT    = ESC + b'\x61\x00'
CMD_ALIGN_CENTER  = ESC + b'\x61\x01'
CMD_BOLD_ON       = ESC + b'\x45\x01'
CMD_BOLD_OFF      = ESC + b'\x45\x00'
CMD_SIZE_NORMAL   = GS  + b'!\x00'
CMD_SIZE_DOUBLE   = GS  + b'!\x11'   # double width + height
CMD_SIZE_QUAD     = GS  + b'!\x33'   # 4× width + height
CMD_CUT           = GS  + b'V\x00'
CMD_FEED          = b'\n'

SEP = b'-' * 32   # ASCII separator, safe for any code page

# ── Settings persistence ───────────────────────────────────────────────────────
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {'last_order': 0, 'printer': ''}

def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as fh:
        json.dump(settings, fh, ensure_ascii=False, indent=2)

# ── Printer helpers ────────────────────────────────────────────────────────────
def list_printers() -> list[str]:
    if not HAS_WIN32:
        return []
    flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
    return [p[2] for p in win32print.EnumPrinters(flags)]

def encode_text(text: str) -> bytes:
    return text.encode('cp866', errors='replace')

def build_receipt(order_number: int) -> bytes:
    now      = datetime.now()
    date_str = now.strftime('%d.%m.%Y')
    time_str = now.strftime('%H:%M:%S')

    data  = CMD_INIT
    data += CMD_CODEPAGE_866       # tell printer to use CP866 (Cyrillic)
    data += CMD_ALIGN_CENTER
    data += CMD_FEED
    data += CMD_BOLD_ON
    data += encode_text('ЗАМОВЛЕННЯ') + CMD_FEED
    data += CMD_BOLD_OFF
    data += SEP + CMD_FEED
    data += CMD_SIZE_QUAD
    data += CMD_BOLD_ON
    data += encode_text(str(order_number)) + CMD_FEED
    data += CMD_BOLD_OFF
    data += CMD_SIZE_NORMAL
    data += SEP + CMD_FEED
    data += encode_text(date_str) + CMD_FEED   # date on its own line
    data += encode_text(time_str) + CMD_FEED   # time on its own line
    data += CMD_FEED * 3
    data += CMD_CUT
    return data

def raw_print(printer_name: str, data: bytes) -> None:
    if not HAS_WIN32:
        raise RuntimeError(
            'Бібліотека pywin32 не встановлена.\n'
            'Виконайте: pip install pywin32'
        )
    hprinter = win32print.OpenPrinter(printer_name)
    try:
        win32print.StartDocPrinter(hprinter, 1, ('Order', None, 'RAW'))
        try:
            win32print.StartPagePrinter(hprinter)
            win32print.WritePrinter(hprinter, data)
            win32print.EndPagePrinter(hprinter)
        finally:
            win32print.EndDocPrinter(hprinter)
    finally:
        win32print.ClosePrinter(hprinter)

# ── Main application ───────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Друк замовлень — XPrinter')
        self.resizable(False, False)
        self.configure(bg='#f0f0f0')
        self.settings = load_settings()
        self._build_ui()
        self._refresh_printers()
        self._update_counter_labels()
        # Bind Enter key to manual print
        self.bind('<Return>', lambda _: self._print_manual())

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        PAD = {'padx': 12, 'pady': 6}

        # ── Printer selector ──────────────────────────────────────────────────
        frm_printer = ttk.LabelFrame(self, text=' Принтер ', padding=10)
        frm_printer.grid(row=0, column=0, sticky='ew', **PAD)

        self.printer_var = tk.StringVar(value=self.settings.get('printer', ''))
        self.printer_combo = ttk.Combobox(
            frm_printer, textvariable=self.printer_var,
            width=38, state='readonly', font=('Segoe UI', 10)
        )
        self.printer_combo.pack(side='left', fill='x', expand=True)

        ttk.Button(
            frm_printer, text='↺ Оновити',
            command=self._refresh_printers
        ).pack(side='left', padx=(6, 0))

        # ── Manual order number ───────────────────────────────────────────────
        frm_manual = ttk.LabelFrame(self, text=' Ввести номер вручну ', padding=10)
        frm_manual.grid(row=1, column=0, sticky='ew', **PAD)

        vcmd = (self.register(lambda s: s == '' or (s.isdigit() and len(s) <= 6)), '%P')
        self.manual_var = tk.StringVar()
        self.entry = ttk.Entry(
            frm_manual, textvariable=self.manual_var,
            width=12, font=('Segoe UI', 28, 'bold'), justify='center',
            validate='key', validatecommand=vcmd
        )
        self.entry.pack(side='left', padx=(0, 10))
        self.entry.focus_set()

        ttk.Button(
            frm_manual, text='🖨  Друкувати',
            command=self._print_manual
        ).pack(side='left', ipady=8, ipadx=10)

        # ── Auto counter ──────────────────────────────────────────────────────
        frm_auto = ttk.LabelFrame(self, text=' Автоматичний лічильник ', padding=10)
        frm_auto.grid(row=2, column=0, sticky='ew', **PAD)

        inner = ttk.Frame(frm_auto)
        inner.pack()

        ttk.Label(inner, text='Надруковано:', font=('Segoe UI', 10)).grid(
            row=0, column=0, padx=(0, 4), sticky='e'
        )
        self.lbl_last = ttk.Label(
            inner, text='0', font=('Segoe UI', 22, 'bold'), width=5, anchor='center'
        )
        self.lbl_last.grid(row=0, column=1, padx=6)

        ttk.Label(inner, text='Наступний:', font=('Segoe UI', 10)).grid(
            row=0, column=2, padx=(10, 4), sticky='e'
        )
        self.lbl_next = ttk.Label(
            inner, text='1', font=('Segoe UI', 22, 'bold'),
            width=5, anchor='center', foreground='#2a7d2a'
        )
        self.lbl_next.grid(row=0, column=3, padx=6)

        btn_next = ttk.Button(
            frm_auto, text='🖨  Друкувати наступний  (+1)',
            command=self._print_next
        )
        btn_next.pack(fill='x', pady=(10, 0), ipady=10)

        btn_reset = ttk.Button(
            frm_auto, text='Скинути лічильник до 0',
            command=self._reset_counter
        )
        btn_reset.pack(fill='x', pady=(6, 0))

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value='Готово до роботи')
        lbl_status = ttk.Label(
            self, textvariable=self.status_var,
            relief='sunken', anchor='w', font=('Segoe UI', 9),
            padding=(6, 2)
        )
        lbl_status.grid(row=3, column=0, sticky='ew', padx=12, pady=(4, 10))

    # ── Actions ────────────────────────────────────────────────────────────────
    def _refresh_printers(self) -> None:
        printers = list_printers()
        if not printers and not HAS_WIN32:
            printers = ['(демо — pywin32 не встановлено)']
        self.printer_combo['values'] = printers
        saved = self.settings.get('printer', '')
        if saved in printers:
            self.printer_var.set(saved)
        elif printers:
            self.printer_var.set(printers[0])

    def _update_counter_labels(self) -> None:
        last = self.settings.get('last_order', 0)
        self.lbl_last.config(text=str(last))
        self.lbl_next.config(text=str(last + 1))

    def _selected_printer(self) -> str | None:
        p = self.printer_var.get()
        if not p or '(демо' in p:
            messagebox.showerror('Помилка', 'Оберіть принтер зі списку.')
            return None
        return p

    def _do_print(self, order_number: int) -> bool:
        printer = self._selected_printer()
        if not printer:
            return False
        try:
            raw_print(printer, build_receipt(order_number))
            self.settings['printer'] = printer
            save_settings(self.settings)
            ts = datetime.now().strftime('%H:%M:%S')
            self.status_var.set(f'✓  Надруковано замовлення №{order_number}  [{ts}]')
            return True
        except Exception as exc:
            messagebox.showerror('Помилка друку', str(exc))
            self.status_var.set(f'✗  Помилка: {exc}')
            return False

    def _print_manual(self) -> None:
        val = self.manual_var.get().strip()
        if not val:
            messagebox.showwarning('Увага', 'Введіть номер замовлення.')
            return
        self._do_print(int(val))

    def _print_next(self) -> None:
        next_num = self.settings.get('last_order', 0) + 1
        if self._do_print(next_num):
            self.settings['last_order'] = next_num
            save_settings(self.settings)
            self._update_counter_labels()

    def _reset_counter(self) -> None:
        if messagebox.askyesno('Скидання лічильника', 'Скинути лічильник до 0?'):
            self.settings['last_order'] = 0
            save_settings(self.settings)
            self._update_counter_labels()
            self.status_var.set('Лічильник скинуто до 0')


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if sys.platform != 'win32':
        # Allow opening on non-Windows for UI testing
        import tkinter.messagebox as mb
        root = tk.Tk()
        root.withdraw()
        mb.showwarning(
            'Windows only',
            'Ця програма призначена для Windows.\n'
            'Друк недоступний на поточній платформі.'
        )
        root.destroy()

    app = App()
    app.mainloop()
