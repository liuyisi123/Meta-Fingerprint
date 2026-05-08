"""Dark medical-grade stylesheet for Meta-Fingerprint Monitor."""

PALETTE = {
    "bg_deep":    "#060E1A",
    "bg_base":    "#0A1628",
    "bg_card":    "#0D2137",
    "bg_panel":   "#112845",
    "bg_hover":   "#1A3A5C",
    "accent":     "#00C8F0",
    "accent2":    "#0088CC",
    "success":    "#00E676",
    "warning":    "#FFB300",
    "danger":     "#FF4444",
    "text_prim":  "#E8F4FD",
    "text_sec":   "#7AAFCF",
    "text_mute":  "#3A6A8A",
    "border":     "#1A3A5C",
    "border_acc": "#00C8F0",
    "ecg_col":    "#00FF88",
    "ppg_col":    "#00C8F0",
    "abp_col":    "#FF6B35",
}

QSS = """
/* ───────────────────────── Global ───────────────────────── */
QWidget {
    background-color: %(bg_base)s;
    color: %(text_prim)s;
    font-family: "Segoe UI", "SF Pro Display", Arial, sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: %(bg_deep)s; }
QStatusBar  { background-color: %(bg_deep)s; color: %(text_sec)s; font-size: 11px; border-top: 1px solid %(border)s; }
QStatusBar::item { border: none; }

/* ───────────────────────── Sidebar ──────────────────────── */
#sidebar {
    background-color: %(bg_deep)s;
    border-right: 1px solid %(border)s;
}
#sidebar QPushButton {
    background: transparent;
    color: %(text_sec)s;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}
#sidebar QPushButton:hover {
    background-color: %(bg_hover)s;
    color: %(text_prim)s;
}
#sidebar QPushButton:checked {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 %(accent2)s, stop:1 %(bg_hover)s);
    color: #FFFFFF;
    border-left: 3px solid %(accent)s;
}
#logo_label {
    color: %(accent)s;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 1px;
}
#version_label { color: %(text_mute)s; font-size: 10px; }

/* ───────────────────────── TopBar ───────────────────────── */
#topbar {
    background-color: %(bg_base)s;
    border-bottom: 1px solid %(border)s;
    min-height: 50px;
}
#page_title {
    color: %(text_prim)s;
    font-size: 18px;
    font-weight: 600;
}

/* ───────────────────────── Cards ────────────────────────── */
.MetricCard {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    padding: 16px;
}
.MetricCard:hover { border-color: %(accent)s; }
#card_value {
    color: %(accent)s;
    font-size: 28px;
    font-weight: 700;
}
#card_label  { color: %(text_sec)s; font-size: 11px; letter-spacing: 0.5px; }
#card_unit   { color: %(text_mute)s; font-size: 11px; }
#card_trend_up   { color: %(success)s; font-size: 11px; }
#card_trend_down { color: %(danger)s;  font-size: 11px; }

/* ─────────────────────── Buttons ────────────────────────── */
QPushButton#btn_primary {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 %(accent2)s, stop:1 %(accent)s);
    color: #000000;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 700;
    font-size: 13px;
}
QPushButton#btn_primary:hover { background: %(accent)s; }
QPushButton#btn_primary:pressed { background: %(accent2)s; }
QPushButton#btn_danger {
    background-color: %(danger)s;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
}
QPushButton#btn_secondary {
    background-color: transparent;
    color: %(accent)s;
    border: 1px solid %(accent)s;
    border-radius: 8px;
    padding: 9px 22px;
    font-weight: 500;
}
QPushButton#btn_secondary:hover { background-color: %(bg_hover)s; }
QPushButton#btn_icon {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px;
    color: %(text_sec)s;
    font-size: 16px;
}
QPushButton#btn_icon:hover { color: %(accent)s; background: %(bg_hover)s; }

/* ─────────────────────── Inputs ─────────────────────────── */
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: %(bg_panel)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 8px 12px;
    color: %(text_prim)s;
    selection-background-color: %(accent2)s;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border-color: %(accent)s;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { color: %(accent)s; }
QComboBox QAbstractItemView {
    background-color: %(bg_panel)s;
    border: 1px solid %(border)s;
    selection-background-color: %(bg_hover)s;
}

/* ─────────────────────── Table ──────────────────────────── */
QTableWidget {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    gridline-color: %(border)s;
    alternate-background-color: %(bg_panel)s;
}
QTableWidget::item { padding: 10px 12px; border: none; }
QTableWidget::item:selected {
    background-color: %(bg_hover)s;
    color: %(text_prim)s;
}
QHeaderView::section {
    background-color: %(bg_deep)s;
    color: %(text_sec)s;
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid %(accent2)s;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ─────────────────────── Tabs ───────────────────────────── */
QTabWidget::pane {
    border: 1px solid %(border)s;
    border-radius: 8px;
    background-color: %(bg_card)s;
}
QTabBar::tab {
    background: %(bg_panel)s;
    color: %(text_sec)s;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: %(bg_card)s; color: %(accent)s; border-top: 2px solid %(accent)s; }

/* ─────────────────────── Scrollbar ──────────────────────── */
QScrollBar:vertical {
    background: %(bg_deep)s; width: 6px; border: none;
}
QScrollBar::handle:vertical {
    background: %(text_mute)s; border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ─────────────────────── Progress ───────────────────────── */
QProgressBar {
    background-color: %(bg_panel)s;
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 %(accent2)s, stop:1 %(accent)s);
    border-radius: 4px;
}

/* ─────────────────────── Slider ─────────────────────────── */
QSlider::groove:horizontal {
    background: %(bg_panel)s; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: %(accent)s; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }

/* ─────────────────────── Misc ───────────────────────────── */
QSplitter::handle { background: %(border)s; width: 1px; height: 1px; }
QGroupBox {
    border: 1px solid %(border)s;
    border-radius: 8px;
    margin-top: 8px;
    padding-top: 8px;
    color: %(text_sec)s;
    font-size: 11px;
    letter-spacing: 0.5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px; top: -1px;
    background-color: %(bg_card)s;
    padding: 0 6px;
    color: %(accent)s;
    text-transform: uppercase;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QToolTip {
    background-color: %(bg_panel)s;
    color: %(text_prim)s;
    border: 1px solid %(accent)s;
    border-radius: 4px;
    padding: 4px 8px;
}
#separator { background: %(border)s; max-height: 1px; min-height: 1px; }
""" % PALETTE
