"""
Decomposition Tab - Manages EMG signal decomposition.
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, 
    QCheckBox, QStackedWidget, QSizePolicy, QFrame,
     QMessageBox, QSplitter, QSpinBox, QDoubleSpinBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEventLoop

# Visualization
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from scd_app.gui.style.styling import (
    COLORS, FONT_SIZES, SPACING, FONT_FAMILY,
    get_section_header_style, get_label_style, get_button_style
)

from scd_app.core.config import SessionConfig
import torch
from scd_app.core.decomp_worker import DecompositionWorker

class DecompositionTab(QWidget):
    """Decomposition tab for EMG signal decomposition."""
    
    # Signal emits the decomp file path so Edition tab can load it
    decomposition_complete = pyqtSignal(Path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.config: Optional[SessionConfig] = None
        self.emg_path: Optional[Path] = None
        self.grid_configs: Dict = {}
        self.worker = None
        
        # EMG data
        self.emg_data = None
        self.sampling_rate = 2048
        self.rejected_channels = []
        self.plateau_coords = None
        
        # UI References
        self.grid_selector = None
        self.param_stack = None
        self.param_widgets = {}
        
        # Matplotlib cleanup
        self.cid = None

        # Time window selection state
        self.sel_start: float = 0.0
        self.sel_end = None
        self._selection_state: str = 'idle'   # 'idle', 'start_set', 'complete'
        self._rms_ax = None
        self._line_start = None
        self._line_end = None
        self._canvas_click_cid = None

        self.init_ui()

    def setup_session(self, config: SessionConfig, emg_paths: list):
        """Called by MainWindow when Configuration is applied."""
        self.config = config
        self.emg_paths = [Path(p) if not isinstance(p, Path) else p for p in emg_paths]
        self.emg_path = self.emg_paths[0]
        self.sampling_rate = config.sampling_frequency
        
        n_files = len(self.emg_paths)
        self.file_path_label.setText(
            f"\U0001f4c4 {self.emg_path.name}" if n_files == 1 
            else f"\U0001f4c4 {n_files} files (first: {self.emg_path.name})"
        )
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px; font-weight: bold;"
        )
        
        self._load_emg_data()
        self._load_grid_configs()

        # Show a ready prompt — channel rejection + RMS happen when user clicks Start
        self.figure.clf()
        self.figure.set_facecolor(COLORS['background'])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        ax.text(0.5, 0.5,
                "Session loaded.\nClick  'Start Decomposition'  to begin.",
                color=COLORS['text_muted'], fontsize=14,
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()

        print(f"Decomposition Tab Ready: {len(self.grid_configs)} grids, {n_files} file(s).")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        
        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 20)
        layout.setSpacing(4)
        
        # === Global parameters (all grids) ===
        layout.addWidget(QLabel("GLOBAL PARAMETERS", styleSheet=get_section_header_style('info')))
        
        global_grid = QGridLayout()
        global_grid.setSpacing(2)
        self.global_widgets = {}
        row = 0
        
        def add_global(label, widget, key):
            nonlocal row
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['info_light']};")
            global_grid.addWidget(lbl, row, 0)
            global_grid.addWidget(widget, row, 1)
            self.global_widgets[key] = widget
            row += 1
        
        add_global("SIL Threshold:", QLineEdit("0.85"), "sil_threshold")
        add_global("Iterations:", QLineEdit("20"), "iterations")
        
        clamp = QComboBox()
        clamp.addItems(["True", "False"])
        add_global("Clamping:", clamp, "clamp")
        
        fitness = QComboBox()
        fitness.addItems(["CoV", "SIL"])
        add_global("Fitness:", fitness, "fitness")
        
        peel = QComboBox()
        peel.addItems(["True", "False"])
        add_global("Peel Off:", peel, "peel_off")
        
        swarm = QComboBox()
        swarm.addItems(["True", "False"])
        add_global("Swarm:", swarm, "swarm")
        
        muap_win_spin = QDoubleSpinBox()
        muap_win_spin.setRange(1, 200)
        muap_win_spin.setSingleStep(1)
        muap_win_spin.setValue(20)
        muap_win_spin.setSuffix(" ms")
        add_global("MUAP Window:", muap_win_spin, "muap_window_ms")
        
        layout.addLayout(global_grid)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep)
    
        # === Batch options ===
        batch_label = QLabel("BATCH OPTIONS", styleSheet=get_section_header_style('info'))
        layout.addWidget(batch_label)
        
        batch_grid = QGridLayout()
        batch_grid.setSpacing(2)
        
        lbl1 = QLabel("Channel Rejection:")
        lbl1.setStyleSheet(f"color: {COLORS['info_light']};")
        self.rejection_mode = QComboBox()
        self.rejection_mode.addItems(["Per file", "First file only"])
        batch_grid.addWidget(lbl1, 0, 0)
        batch_grid.addWidget(self.rejection_mode, 0, 1)
        
        lbl2 = QLabel("Time Window:")
        lbl2.setStyleSheet(f"color: {COLORS['info_light']};")
        self.time_mode = QComboBox()
        self.time_mode.addItems(["Manual selection", "Full file"])
        self.time_mode.currentTextChanged.connect(self._on_time_mode_changed)
        batch_grid.addWidget(lbl2, 1, 0)
        batch_grid.addWidget(self.time_mode, 1, 1)

        layout.addLayout(batch_grid)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep2)

        # === Per-grid parameters ===
        layout.addWidget(QLabel("PER-GRID PARAMETERS", styleSheet=get_section_header_style('info')))
        
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Select Grid:"))
        self.grid_selector = QComboBox()
        self.grid_selector.currentIndexChanged.connect(self._on_grid_changed)
        sel_layout.addWidget(self.grid_selector)
        layout.addLayout(sel_layout)
        
        self.param_stack = QStackedWidget()
        layout.addWidget(self.param_stack)
        
        layout.addStretch()

        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Decomposition")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_decomposition)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {COLORS['success']}, stop:1 #2F855A);
                color: white; border-radius: 6px; font-weight: bold; 
                font-size: {FONT_SIZES['medium']};
            }}
            QPushButton:hover {{ background-color: #48BB78; }}
            QPushButton:pressed {{ background-color: #276749; }}
            QPushButton:disabled {{ 
                background-color: {COLORS['background_input']}; 
                color: {COLORS['text_muted']}; 
            }}
        """)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_decomposition)
        self.stop_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-radius: 6px; font-weight: bold;"
        )
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        return container
    
    def _create_right_panel(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(
            f"background-color: {COLORS['background']}; "
            f"border-left: 2px solid {COLORS['border']};"
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setAlignment(Qt.AlignLeft)
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10pt; "
            f"padding: 5px; margin: 5px;"
        )
        
        self.grid_indicator_label = QLabel("")
        self.grid_indicator_label.setStyleSheet(
            f"color: {COLORS['info']}; font-size: 10pt; font-weight: bold; "
            f"padding: 5px; margin: 5px;"
        )
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.file_path_label)
        header_layout.addWidget(self.grid_indicator_label)
        header_layout.addStretch()
        layout.addLayout(header_layout, stretch=0)

        # --- Time window bar (shown below header when Manual selection) ---
        _edit_style = (
            f"QLineEdit {{ background-color: {COLORS.get('background_input','#33334d')};"
            f" color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};"
            f" border-radius: 3px; padding: 2px 6px;"
            f" font-size: {FONT_SIZES.get('small','9pt')}; }}"
        )
        _ts_btn_style = (
            f"QPushButton {{ background-color: {COLORS.get('background_input','#33334d')};"
            f" color: {COLORS['foreground']}; border: 1px solid {COLORS['border']};"
            f" border-radius: 4px; padding: 3px 10px;"
            f" font-size: {FONT_SIZES.get('small','9pt')}; }}"
            f"QPushButton:hover {{ border-color: {COLORS.get('info','#4a9eff')}; }}"
        )
        self.time_sel_widget = QWidget()
        ts_bar = QHBoxLayout(self.time_sel_widget)
        ts_bar.setContentsMargins(4, 0, 4, 4)
        ts_bar.setSpacing(6)

        lbl_s = QLabel("Start (s):")
        lbl_s.setStyleSheet(f"color: {COLORS['info_light']}; font-size: {FONT_SIZES.get('small','9pt')};")
        self.start_time_edit = QLineEdit("0.00")
        self.start_time_edit.setStyleSheet(_edit_style)
        self.start_time_edit.setFixedWidth(72)
        self.start_time_edit.editingFinished.connect(self._on_start_time_entered)

        lbl_e = QLabel("End (s):")
        lbl_e.setStyleSheet(f"color: {COLORS['info_light']}; font-size: {FONT_SIZES.get('small','9pt')};")
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText("…")
        self.end_time_edit.setStyleSheet(_edit_style)
        self.end_time_edit.setFixedWidth(72)
        self.end_time_edit.editingFinished.connect(self._on_end_time_entered)

        self.clear_time_btn = QPushButton("✕  Clear")
        self.clear_time_btn.setStyleSheet(_ts_btn_style)
        self.clear_time_btn.clicked.connect(self._on_clear_time_selection)

        instr_lbl = QLabel("  Click plot: 1st = start,  2nd = end")
        instr_lbl.setStyleSheet(
            f"color: {COLORS.get('text_muted','#6c7086')}; font-size: 8pt; font-style: italic;")

        ts_bar.addWidget(lbl_s)
        ts_bar.addWidget(self.start_time_edit)
        ts_bar.addWidget(lbl_e)
        ts_bar.addWidget(self.end_time_edit)
        ts_bar.addWidget(self.clear_time_btn)
        ts_bar.addWidget(instr_lbl)
        ts_bar.addStretch()

        layout.addWidget(self.time_sel_widget, stretch=0)
        self.time_sel_widget.setVisible(False)  # shown only during time window selection

        self.figure = Figure(facecolor=COLORS['background'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLORS['background'])
        self.ax.text(
            0.5, 0.5, "Waiting for configuration...", 
            color=COLORS['text_muted'], 
            ha='center', va='center', transform=self.ax.transAxes
        )
        self.ax.axis('off')
        
        layout.addWidget(self.canvas, stretch=1)
        return container

    def _load_emg_data(self):
        """Load EMG data using the layout from config."""
        import copy
        from scd_app.io.data_loader import load_field

        try:
            layout = getattr(self.config, 'data_layout', None)
            if layout is None:
                raise ValueError("No data layout configured -- select a Data Format in Configuration tab")

            # Strip layout-level channel slice so all channels are loaded.
            # Per-grid channel selection is done downstream via port.electrode.channels
            # (absolute indices into the full array).
            layout_full = copy.deepcopy(layout)
            layout_full["fields"]["emg"].pop("channels", None)

            emg = load_field(self.emg_path, layout_full, "emg")
            # load_field returns (samples, channels) as torch.Tensor
            self.emg_data = emg
            print(f"Loaded EMG data: {self.emg_data.shape}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load EMG data:\n{str(e)}")
            print(f"Error loading EMG data: {e}")

    def _load_grid_configs(self):
        """Populate grid configurations based on SessionConfig."""
        self.grid_configs.clear()
        self.grid_selector.clear()
        
        while self.param_stack.count():
            w = self.param_stack.widget(0)
            self.param_stack.removeWidget(w)
            w.deleteLater()
            
        if not self.config:
            return

        for port in self.config.ports:
            if not port.enabled:
                continue
            
            n_channels = len(port.electrode.channels)
            extension_factor = int(np.ceil(1000 / n_channels))
            
            # Set filter defaults based on electrode type
            is_surface = port.electrode.type == "surface"
            lowpass = 500 if is_surface else 4400
            highpass = 10
            
            defaults = {
                "extension_factor": extension_factor,
                "lowpass_hz": lowpass,
                "highpass_hz": highpass,
                "notch_filter": "None",
                "notch_harmonics": False,
            }
            
            electrode_type_label = "Surface" if is_surface else "Intramuscular"
            
            self.grid_configs[port.name] = {
                "params": defaults,
                "channels": port.electrode.channels,
                "num_channels": n_channels,
                "electrode_type": port.electrode.name,
                "electrode_class": port.electrode.type,
            }
            
            self.grid_selector.addItem(
                f"{port.name} ({port.electrode.name})", 
                port.name
            )
            
            page = self._create_param_page(port.name, defaults, electrode_type_label)
            self.param_stack.addWidget(page)

    def _create_param_page(self, port_name: str, defaults: dict, electrode_type: str = "") -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(10)
        
        widgets = {}
        row = 0
        
        def add_row(label, widget):
            nonlocal row
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['info_light']};")
            layout.addWidget(lbl, row, 0)
            layout.addWidget(widget, row, 1)
            row += 1
            return widget
        
        # Type indicator (read-only)
        type_label = QLabel(electrode_type if electrode_type else "Unknown")
        type_label.setStyleSheet(f"color: {COLORS['foreground']}; font-weight: bold;")
        add_row("Electrode Type:", type_label)
        
        widgets['extension_factor'] = add_row("Extension Factor:", QLineEdit(str(defaults['extension_factor'])))
        widgets['highpass_hz'] = add_row("High-pass (Hz):", QLineEdit(str(defaults['highpass_hz'])))
        widgets['lowpass_hz'] = add_row("Low-pass (Hz):", QLineEdit(str(defaults['lowpass_hz'])))
        
        notch = QComboBox()
        notch.addItems(["None", "50", "60"])
        notch.setCurrentText(defaults['notch_filter'])
        widgets['notch_filter'] = add_row("Notch Filter:", notch)
        
        harmonics_cb = QCheckBox("Include Harmonics")
        harmonics_cb.setChecked(defaults['notch_harmonics'])
        widgets['notch_harmonics'] = add_row("", harmonics_cb)

        self.param_widgets[port_name] = widgets
        return page
    
    def _on_grid_changed(self, index: int):
        if index >= 0:
            self.param_stack.setCurrentIndex(index)

    def _sync_params_from_ui(self):
        """Update grid_configs with values from global + per-grid UI."""
        # Read global params once
        global_params = {
            "sil_threshold": float(self.global_widgets['sil_threshold'].text()),
            "iterations": int(self.global_widgets['iterations'].text()),
            "clamp": self.global_widgets['clamp'].currentText() == "True",
            "fitness": self.global_widgets['fitness'].currentText(),
            "peel_off": self.global_widgets['peel_off'].currentText() == "True",
            "swarm": self.global_widgets['swarm'].currentText() == "True",
            "muap_window_ms": self.global_widgets['muap_window_ms'].value(),
        }
        
        # Merge into each grid
        for port_name, widgets in self.param_widgets.items():
            try:
                params = self.grid_configs[port_name]["params"]
                
                # Global
                params.update(global_params)
                
                # Per-grid
                params["extension_factor"] = int(widgets['extension_factor'].text())
                params["highpass_hz"] = float(widgets['highpass_hz'].text())
                params["lowpass_hz"] = float(widgets['lowpass_hz'].text())
                params["notch_filter"] = widgets['notch_filter'].currentText()
                params["notch_harmonics"] = widgets['notch_harmonics'].isChecked()
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not sync parameter for {port_name}: {e}")

    # ------------------------------------------------------------------ #
    #  Aux / force channel helpers                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_filename_task(stem: str):
        """Extract (direction, finger_names) from a recording filename stem.

        Looks for:
          mvc-<number><ext|flex>   → direction = "ext" | "flex"
          fing-<LETTERS>           → finger codes → full names
                                     T=thumb  I=index  M=middle  R=ring  L=little

        Returns (direction, [finger_names]) or (None, []) if not parseable.
        """
        import re
        _FINGER = {'T': 'thumb', 'I': 'index', 'M': 'middle',
                   'R': 'ring',  'L': 'little'}
        direction = None
        fingers   = []

        mvc_m = re.search(r'mvc-([^_]+)', stem, re.IGNORECASE)
        if mvc_m:
            mvc_str = mvc_m.group(1).lower()
            if 'ext' in mvc_str:
                direction = 'ext'
            elif 'flex' in mvc_str:
                direction = 'flex'

        fing_m = re.search(r'fing-([A-Za-z]+)', stem)
        if fing_m:
            fingers = [_FINGER[c] for c in fing_m.group(1).upper() if c in _FINGER]

        return direction, fingers

    @staticmethod
    def _select_aux_for_task(aux_cfgs: list, direction, fingers: list) -> list:
        """Decide which aux channels to overlay according to the three cases:

        Case 2 – no aux configs at all         → return []  (nothing to show)
        Case 3 – aux configs have no unit label → return all configs
        Case 1 – aux configs have unit labels   → return those whose label
                  contains the finger name AND direction (case-insensitive);
                  if none match, fall back to showing all labelled ones.
        """
        if not aux_cfgs:
            return []                                        # Case 2

        labeled   = [a for a in aux_cfgs if a.get('unit', '').strip()]
        unlabeled = [a for a in aux_cfgs if not a.get('unit', '').strip()]

        if not labeled:
            return aux_cfgs                                  # Case 3 – no labels

        if direction is None or not fingers:
            return labeled                                   # can't match → show all

        matched = [
            a for a in labeled
            if any(f in a['unit'].lower() for f in fingers)
            and direction in a['unit'].lower()
        ]
        return matched if matched else labeled               # Case 1 (fallback = all)

    def _manual_channel_rejection(self):
        """Show EMG channels per grid and let user select which to remove."""
        if self.emg_data is None:
            return
        
        self._cleanup_matplotlib_widgets()
        self.figure.set_facecolor(COLORS['background'])
        
        # Build grid list
        grid_list = list(self.grid_configs.items())
        if not grid_list:
            return
        
        # Init rejection masks
        self.rejected_channels = []
        for port_idx, (port_name, config) in enumerate(grid_list):
            n_channels = len(config['channels'])
            if port_idx < len(self.rejected_channels):
                pass  # keep existing
            else:
                self.rejected_channels.append(np.zeros(n_channels, dtype=int))
        
        # ── Aux / force overlay setup ─────────────────────────────────────
        _aux_cfgs   = getattr(self.config, 'aux_channels', []) if self.config else []
        _stem       = self.emg_path.stem if self.emg_path else ""
        _direction, _fingers = self._parse_filename_task(_stem)
        _aux_to_show = self._select_aux_for_task(_aux_cfgs, _direction, _fingers)

        if _aux_to_show:
            print(f"  [force overlay] direction={_direction!r}  "
                  f"fingers={_fingers}  channels={[a.get('unit') for a in _aux_to_show]}")

        import time
        nav = {'current': 0, 'show_force': bool(_aux_to_show)}

        def draw_grid(grid_idx):
            """Draw a single grid's channels."""
            self.figure.clf()
            
            port_name, config = grid_list[grid_idx]
            channels = config['channels']
            n_channels = len(channels)
            mask = self.rejected_channels[grid_idx]

            ax = self.figure.add_axes([0.03, 0.14, 0.94, 0.80])
            ax.set_facecolor(COLORS['background'])
            self.figure.set_facecolor(COLORS['background'])

            # Title
            self.figure.text(
                0.5, 0.96,
                f"{port_name}  ({grid_idx + 1}/{len(grid_list)})",
                ha="center", va="center",
                fontsize=13, weight='bold', color=COLORS['foreground'],
            )

            # Instructions
            self.figure.text(
                0.5, 0.08,
                "Click = Toggle channel | Scroll = Zoom time | Shift+Scroll = Zoom channels | Right-drag = Pan | R = Reset",
                ha="center", va="center",
                fontsize=10, weight='bold', color=COLORS['info'],
            )

            grid_data = self.emg_data[:, channels].numpy()
            step = 1
            disp_data = grid_data
            max_len = disp_data.shape[0]

            # Normalise separation using only active (non-rejected) channels
            active_idx = np.where(mask == 0)[0]
            ref_data = disp_data[:, active_idx] if len(active_idx) > 0 else disp_data
            active_std = np.std(ref_data)
            separation = active_std * 15 if active_std > 0 else 1.0

            for ch in range(n_channels):
                is_rejected = mask[ch] == 1
                y_pos = ch * separation
                if is_rejected:
                    # Faint dashed flat line — visually removed but still clickable
                    ax.plot([0, max_len], [y_pos, y_pos],
                            color=COLORS['error'], alpha=0.25, linewidth=0.8,
                            linestyle='--')
                else:
                    ax.plot(disp_data[:, ch] + y_pos,
                            color=COLORS['info'], alpha=0.8, linewidth=1.0)

            # Channel labels on the left
            for ch in range(n_channels):
                is_rejected = mask[ch] == 1
                ax.text(
                    -max_len * 0.01, ch * separation,
                    f"{ch}", color=COLORS['error'] if is_rejected else COLORS['text_muted'],
                    fontsize=7, ha='right', va='center',
                )
            
            total_height = n_channels * separation
            ax.set_xlim(0, max_len)
            ax.set_ylim(-separation, total_height)
            ax.margins(0)

            # Hide everything except the bottom time axis
            ax.set_yticks([])
            for spine in ('top', 'right', 'left'):
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color(COLORS.get('text_muted', '#888888'))
            ax.tick_params(axis='x', colors=COLORS.get('text_muted', '#888888'),
                           labelsize=8, length=4)

            # Format x-axis ticks as time (seconds), accounting for downsampling
            def _time_fmt(x, pos):
                seconds = x * step / self.sampling_rate
                if seconds >= 60:
                    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
                return f"{seconds:.1f}s"

            ax.xaxis.set_major_formatter(FuncFormatter(_time_fmt))

            # ── Force / aux overlay ───────────────────────────────────────────
            if _aux_to_show and nav.get('show_force', False):
                total_height = n_channels * separation
                force_color  = '#FFD700'   # gold — visible on dark background
                first_label  = True
                for a in _aux_to_show:
                    s = int(a.get('start_chan', 0))
                    e = int(a.get('end_chan',   s + 1))
                    n_total_ch = self.emg_data.shape[1]
                    if s >= e or e > n_total_ch:
                        continue
                    raw = self.emg_data[:, s:e].numpy().mean(axis=1)   # (samples,)
                    sig = raw - raw.mean()                              # remove DC offset
                    peak = max(abs(sig.max()), abs(sig.min()), 1e-9)
                    # Scale to span ~70 % of total channel height, centred
                    sig_scaled = sig / peak * (0.35 * total_height) + total_height / 2
                    label = a.get('unit', a.get('name', 'force')) if first_label else '_'
                    ax.plot(sig_scaled, color=force_color, alpha=0.55,
                            linewidth=1.2, zorder=3, label=label)
                    first_label = False
                ax.legend(
                    loc='upper right', fontsize=7,
                    facecolor=COLORS['background_light'],
                    labelcolor=force_color, framealpha=0.7,
                )

            # --- Navigation buttons ---
            prev_ax = self.figure.add_axes([0.05, 0.01, 0.12, 0.05])
            next_ax = self.figure.add_axes([0.83, 0.01, 0.12, 0.05])
            confirm_ax = self.figure.add_axes([0.4, 0.01, 0.2, 0.05])
            
            is_last = (grid_idx == len(grid_list) - 1)
            is_first = (grid_idx == 0)

            prev_btn = Button(prev_ax, "<- Previous",
                              color=COLORS['background_light'], hovercolor=COLORS['background_hover'])
            prev_btn.label.set_color(COLORS['foreground'] if not is_first else COLORS['text_muted'])
            prev_btn.label.set_fontsize(10)
            
            next_btn = Button(next_ax, "Next  →",
                              color=COLORS['info'] if not is_last else COLORS['background_light'],
                              hovercolor=COLORS['info_light'])
            next_btn.label.set_color('white' if not is_last else COLORS['text_muted'])
            next_btn.label.set_fontsize(11)
            next_btn.label.set_weight('bold')

            confirm_btn = Button(confirm_ax, "CONFIRM ALL",
                                 color=COLORS['success'], hovercolor='#2EA043')
            confirm_btn.label.set_color("white")
            confirm_btn.label.set_weight('bold')
            confirm_btn.label.set_fontsize(10)

            # ── "Show force" checkbox (only when aux channels available) ────
            if _aux_to_show:
                from matplotlib.widgets import CheckButtons
                cb_ax = self.figure.add_axes([0.20, 0.005, 0.13, 0.06])
                cb_ax.set_facecolor(COLORS['background'])
                check = CheckButtons(
                    cb_ax, ['Show force'], [nav.get('show_force', True)]
                )
                check.labels[0].set_color(COLORS['foreground'])
                check.labels[0].set_fontsize(9)
                for r in getattr(check, 'rectangles', []):
                    r.set_edgecolor(COLORS.get('text_muted', '#888'))
                    r.set_facecolor(COLORS['background_light'])

                def on_force_toggle(_label, _check=check):
                    nav['show_force'] = _check.get_status()[0]
                    disconnect()
                    draw_grid(nav['current'])

                check.on_clicked(on_force_toggle)
                nav.setdefault('buttons', []).append(check)

            # Rejected count for this grid
            n_rej = np.sum(mask)
            rej_text = self.figure.text(
                0.5, 0.04,
                f"{n_rej} channel{'s' if n_rej != 1 else ''} rejected" if n_rej > 0 else "",
                ha="center", va="center",
                fontsize=9, color=COLORS['error'] if n_rej > 0 else COLORS['text_muted'],
            )

            # --- Interaction state ---
            state = {
                'press_event': None,
                'last_scroll_time': 0,
                'panning': False,
                'pan_start': None,
                'pan_xlim': None,
                'pan_ylim': None,
            }
            CLICK_TOLERANCE_PX = 5
            SCROLL_GUARD_MS = 300

            def on_press(event):
                if event.inaxes != ax:
                    return
                if event.button == 1:
                    state['press_event'] = event
                elif event.button == 3:
                    state['panning'] = True
                    state['pan_start'] = (event.x, event.y)
                    state['pan_xlim'] = ax.get_xlim()
                    state['pan_ylim'] = ax.get_ylim()

            def on_motion(event):
                if not state['panning'] or event.inaxes != ax:
                    return
                dx = event.x - state['pan_start'][0]
                dy = event.y - state['pan_start'][1]
                xlim = state['pan_xlim']
                ylim = state['pan_ylim']
                bbox = ax.get_window_extent()
                data_dx = -(dx / bbox.width) * (xlim[1] - xlim[0])
                data_dy = -(dy / bbox.height) * (ylim[1] - ylim[0])
                ax.set_xlim(xlim[0] + data_dx, xlim[1] + data_dx)
                ax.set_ylim(ylim[0] + data_dy, ylim[1] + data_dy)
                self.canvas.draw_idle()

            def on_release(event):
                if event.button == 3:
                    state['panning'] = False
                    return
                if event.button != 1 or state['press_event'] is None:
                    return
                press = state['press_event']
                state['press_event'] = None
                
                elapsed_ms = (time.time() - state['last_scroll_time']) * 1000
                if elapsed_ms < SCROLL_GUARD_MS:
                    return
                if abs(event.x - press.x) > CLICK_TOLERANCE_PX or \
                   abs(event.y - press.y) > CLICK_TOLERANCE_PX:
                    return
                if event.inaxes != ax:
                    return
                
                if event.ydata is None:
                    return

                # Find the closest channel by y-position proximity
                closest_ch = None
                min_dist = float('inf')
                for ch in range(n_channels):
                    d = abs(event.ydata - ch * separation)
                    if d < separation * 0.6 and d < min_dist:
                        min_dist = d
                        closest_ch = ch

                if closest_ch is None:
                    return

                mask[closest_ch] = 1 - mask[closest_ch]
                # Redraw so normalisation updates without the toggled channel
                disconnect()
                draw_grid(nav['current'])

            def on_scroll(event):
                if event.inaxes != ax:
                    return
                state['last_scroll_time'] = time.time()
                scale = 0.85 if event.button == 'up' else 1.18
                shift_held = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
                if shift_held:
                    ylim = ax.get_ylim()
                    ydata = event.ydata
                    ax.set_ylim(ydata - (ydata - ylim[0]) * scale,
                                ydata + (ylim[1] - ydata) * scale)
                else:
                    xlim = ax.get_xlim()
                    xdata = event.xdata
                    ax.set_xlim(xdata - (xdata - xlim[0]) * scale,
                                xdata + (xlim[1] - xdata) * scale)
                self.canvas.draw_idle()

            def on_key(event):
                if event.key in ('r', 'R'):
                    ax.set_xlim(0, max_len)
                    ax.set_ylim(-separation, total_height)
                    self.canvas.draw_idle()

            def go_prev(event):
                if not is_first:
                    disconnect()
                    nav['current'] -= 1
                    draw_grid(nav['current'])

            def go_next(event):
                if not is_last:
                    disconnect()
                    nav['current'] += 1
                    draw_grid(nav['current'])

            def on_confirm(event):
                disconnect()
                self.figure.clf()
                self.figure.set_facecolor(COLORS['background'])
                
                total_rej = sum(np.sum(m) for m in self.rejected_channels)
                self.figure.text(
                    0.5, 0.5,
                    f"Rejected {total_rej} channels across {len(grid_list)} grids",
                    ha="center", va="center",
                    fontsize=14, weight='bold', color=COLORS['success'],
                )
                self.canvas.draw()
                
                print("\nChannel Rejection Summary:")
                for pidx, (pname, _) in enumerate(grid_list):
                    n = np.sum(self.rejected_channels[pidx])
                    print(f"  {pname}: {n} channels rejected")
                
                QTimer.singleShot(100, event_loop.quit)

            # Connect
            cids = [
                self.canvas.mpl_connect('button_press_event', on_press),
                self.canvas.mpl_connect('button_release_event', on_release),
                self.canvas.mpl_connect('motion_notify_event', on_motion),
                self.canvas.mpl_connect('scroll_event', on_scroll),
                self.canvas.mpl_connect('key_press_event', on_key),
            ]
            prev_btn.on_clicked(go_prev)
            next_btn.on_clicked(go_next)
            confirm_btn.on_clicked(on_confirm)
            
            # Store for cleanup
            nav['cids'] = cids
            nav.setdefault('buttons', []).extend([prev_btn, next_btn, confirm_btn])
            
            self.canvas.draw()

        def disconnect():
            # Disconnect canvas events
            for cid in nav.get('cids', []):
                self.canvas.mpl_disconnect(cid)
                
            # Disconnect and remove all navigation buttons
            for btn in nav.get('buttons', []):
                btn.disconnect_events()
                try:
                    btn.ax.remove()
                except Exception:
                    pass  # Ignore if already removed
            nav['buttons'] = []

        event_loop = QEventLoop()
        draw_grid(0)
        event_loop.exec_()

    def _cleanup_matplotlib_widgets(self):
        """Disconnect all active matplotlib event handlers before redrawing the canvas."""
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

        if hasattr(self, 'scroll_cid') and self.scroll_cid is not None:
            self.canvas.mpl_disconnect(self.scroll_cid)
            self.scroll_cid = None

        if self._canvas_click_cid is not None:
            self.canvas.mpl_disconnect(self._canvas_click_cid)
            self._canvas_click_cid = None

    # ------------------------------------------------------------------ #
    #  RMS plot & time-window selection                                   #
    # ------------------------------------------------------------------ #

    def _show_rms_plot(self):
        """Display RMS amplitude on the canvas and restore selection lines."""
        if self.emg_data is None or not self.grid_configs:
            return

        self._cleanup_matplotlib_widgets()
        self._rms_ax = None
        self.figure.clf()
        self.figure.set_facecolor(COLORS['background'])

        total_duration = self.emg_data.shape[0] / self.sampling_rate
        time_axis = np.arange(self.emg_data.shape[0]) / self.sampling_rate

        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        self._rms_ax = ax

        colors = ['#4a9eff', '#a78bfa', '#48BB78', '#F6AD55', '#ff6b9d']
        for idx, (port_name, config) in enumerate(self.grid_configs.items()):
            channels = config['channels']
            channel_data = self.emg_data[:, channels].numpy()

            if idx < len(self.rejected_channels):
                rejected = self.rejected_channels[idx]
                active_channels = np.where(rejected == 0)[0]
                channel_data = channel_data[:, active_channels]

            rms = np.sqrt(np.mean(channel_data ** 2, axis=1))
            window = int(self.sampling_rate * 0.1)
            kernel = np.ones(window) / window
            rms_smooth = np.convolve(rms, kernel, mode='same')
            color = colors[idx % len(colors)]
            ax.plot(time_axis, rms_smooth, color=color, label=port_name,
                    linewidth=2.5, alpha=0.85)

        ax.set_xlabel("Time (s)", color=COLORS['foreground'], fontsize=12, weight='bold')
        ax.set_ylabel("RMS Amplitude", color=COLORS['foreground'], fontsize=12, weight='bold')
        ax.tick_params(colors=COLORS['foreground'])
        ax.set_xlim(0, total_duration)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS.get('background_light', '#2a2a3c'))
        ax.spines['bottom'].set_color(COLORS.get('background_light', '#2a2a3c'))
        if len(self.grid_configs) > 1:
            ax.legend(facecolor=COLORS.get('background_light', '#2a2a3c'),
                      labelcolor=COLORS['foreground'], fontsize=9)

        # Selection lines
        self._line_start = ax.axvline(x=0, color=COLORS.get('success', '#48BB78'),
                                      linestyle='--', linewidth=2, visible=False)
        self._line_end   = ax.axvline(x=total_duration, color=COLORS.get('error', '#f38ba8'),
                                      linestyle='--', linewidth=2, visible=False)

        # Default: full range if nothing selected yet
        if self._selection_state == 'idle':
            self.sel_start = 0.0
            self.sel_end   = total_duration
            self._selection_state = 'complete'
            self.start_time_edit.setText(f"{self.sel_start:.2f}")
            self.end_time_edit.setText(f"{self.sel_end:.2f}")

        self._update_time_visuals()
        self.canvas.draw()

        if self.time_mode.currentText() == "Manual selection":
            self._connect_canvas_click()

    def _on_time_mode_changed(self, text: str):
        is_manual = text == "Manual selection"
        if is_manual and self._rms_ax is not None:
            self._connect_canvas_click()
        else:
            self._disconnect_canvas_click()

    def _connect_canvas_click(self):
        self._disconnect_canvas_click()
        self._canvas_click_cid = self.canvas.mpl_connect(
            'button_press_event', self._on_canvas_click)

    def _disconnect_canvas_click(self):
        if self._canvas_click_cid is not None:
            self.canvas.mpl_disconnect(self._canvas_click_cid)
            self._canvas_click_cid = None

    def _on_canvas_click(self, event):
        if self._rms_ax is None or event.inaxes != self._rms_ax:
            return
        if event.button != 1 or event.xdata is None:
            return
        val = event.xdata
        if self._selection_state in ('idle', 'complete'):
            self.sel_start = val
            self.sel_end   = None
            self._selection_state = 'start_set'
        else:  # start_set
            self.sel_end = val
            if self.sel_end < self.sel_start:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            self._selection_state = 'complete'
        self._update_time_visuals()

    def _on_start_time_entered(self):
        try:
            val = float(self.start_time_edit.text())
            self.sel_start = val
            if self._selection_state == 'idle':
                self._selection_state = 'start_set'
            if self.sel_end is not None and self.sel_start > self.sel_end:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            self._update_time_visuals()
        except ValueError:
            pass

    def _on_end_time_entered(self):
        try:
            val = float(self.end_time_edit.text())
            self.sel_end = val
            if self.sel_start is None:
                self.sel_start = 0.0
            if self.sel_start > self.sel_end:
                self.sel_start, self.sel_end = self.sel_end, self.sel_start
            self._selection_state = 'complete'
            self._update_time_visuals()
        except ValueError:
            pass

    def _on_clear_time_selection(self):
        if self.emg_data is not None:
            total_duration = self.emg_data.shape[0] / self.sampling_rate
            self.sel_start = 0.0
            self.sel_end   = total_duration
            self._selection_state = 'complete'
        else:
            self.sel_start = 0.0
            self.sel_end   = None
            self._selection_state = 'idle'
        self._update_time_visuals()

    def _update_time_visuals(self):
        """Sync text boxes and plot lines with current sel_start / sel_end."""
        if self.sel_start is not None:
            self.start_time_edit.setText(f"{self.sel_start:.2f}")
        if self.sel_end is not None:
            self.end_time_edit.setText(f"{self.sel_end:.2f}")
        else:
            self.end_time_edit.clear()

        if self._line_start is not None:
            if self.sel_start is not None:
                self._line_start.set_xdata([self.sel_start, self.sel_start])
                self._line_start.set_visible(True)
            else:
                self._line_start.set_visible(False)

        if self._line_end is not None:
            if self.sel_end is not None:
                self._line_end.set_xdata([self.sel_end, self.sel_end])
                self._line_end.set_visible(True)
            else:
                self._line_end.set_visible(False)

        if self._rms_ax is not None:
            self.canvas.draw_idle()

    def _start_decomposition(self):
        if not self.config or not self.emg_paths or self.emg_data is None:
            QMessageBox.warning(self, "Error", "No session loaded.")
            return

        # ── Phase 2: user confirmed time window → run current file ───────
        if getattr(self, '_awaiting_time_confirmation', False):
            if not self._use_full_file:
                if self._selection_state != 'complete' or self.sel_start is None or self.sel_end is None:
                    QMessageBox.warning(self, "Time Window",
                        "Please select a time window on the EMG plot\n"
                        "(enter Start/End times or click directly on the plot).")
                    return
                self.plateau_coords = np.array([
                    int(self.sel_start * self.sampling_rate),
                    int(self.sel_end * self.sampling_rate),
                ])
            self._awaiting_time_confirmation = False
            self.start_btn.setText("Start Decomposition")
            self._run_current_file()
            return

        # ── Phase 1: initialise queue and prepare the first file ─────────
        self._sync_params_from_ui()
        self._use_full_file   = self.time_mode.currentText() == "Full file"
        self._share_rejection = self.rejection_mode.currentText() == "First file only"
        self._output_dir = (Path(self.config.output_dir) if self.config.output_dir
                            else self.emg_path.parent)
        self._file_queue = list(self.emg_paths)
        self._file_idx   = 0
        self._prepare_current_file()

    def _prepare_current_file(self):
        """Load file, do channel rejection + RMS, then wait for time window (per file)."""
        from scd_app.io.data_loader import load_field

        if self._file_idx >= len(self._file_queue):
            self.grid_indicator_label.setText("All files complete")
            self._reset_ui_state()
            if hasattr(self, '_last_decomp_path') and self._last_decomp_path:
                self.decomposition_complete.emit(self._last_decomp_path)
            return

        file_path = self._file_queue[self._file_idx]
        n_total = len(self._file_queue)
        self.grid_indicator_label.setText(
            f"File {self._file_idx + 1}/{n_total}: {file_path.name}"
        )

        # Load this file's EMG data
        try:
            layout = getattr(self.config, 'data_layout', None)
            emg = load_field(file_path, layout, "emg")
            self.emg_data = emg
            self.emg_path = file_path
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load {file_path.name}:\n{e}")
            self._file_idx += 1
            self._prepare_current_file()
            return

        # Channel rejection: manual per file, or reuse shared mask
        if self._file_idx == 0 or not self._share_rejection:
            self._manual_channel_rejection()
        else:
            self.rejected_channels = [m.copy() for m in self._shared_mask]

        # Save shared mask after first file's rejection
        if self._file_idx == 0:
            self._shared_mask = [m.copy() for m in self.rejected_channels]

        # Show cleaned RMS plot (reset selection so lines are cleared)
        self._selection_state = 'idle'
        self._show_rms_plot()

        if self._use_full_file:
            self.plateau_coords = np.array([0, self.emg_data.shape[0]])
            self._run_current_file()
        else:
            self.start_btn.setVisible(True)
            self.stop_btn.setVisible(False)
            self._awaiting_time_confirmation = True
            self.start_btn.setText("▶  Confirm & Run")
            self.time_sel_widget.setVisible(True)

    def _run_current_file(self):
        """Start the decomposition worker for the current file."""
        file_path = self._file_queue[self._file_idx]

        self._cleanup_matplotlib_widgets()
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.time_sel_widget.setVisible(False)

        self.figure.clf()
        self.figure.set_facecolor(COLORS['background'])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        ax.text(0.5, 0.5, f"Decomposing {file_path.name}...",
                color=COLORS['warning'], fontsize=16, weight='bold',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()

        save_path = self._output_dir / f"{file_path.stem}_decomp_output.pkl"

        aux_configs = getattr(self.config, "aux_channels", [])
        self.worker = DecompositionWorker(
            self.emg_data, self.grid_configs, self.rejected_channels,
            self.plateau_coords, self.sampling_rate, save_path,
            aux_configs=aux_configs,
        )
        self.worker.progress.connect(self._update_grid_indicator)
        self.worker.finished.connect(self._on_file_decomposition_finished)
        self.worker.error.connect(self._on_decomposition_error)
        self.worker.source_found.connect(self._on_source_found)
        self.worker.start()

    def _on_file_decomposition_finished(self, results):
        """One file done — move to next (which repeats channel rejection + time selection)."""
        decomp_path = Path(results.get('path'))
        self._last_decomp_path = decomp_path

        self._file_idx += 1
        self._prepare_current_file()

    def _update_grid_indicator(self, message: str):
        """Update grid indicator when progress says 'Processing ...'"""
        if message.startswith("Processing "):
            # Extract "Processing GridName (2/6)..."
            self.grid_indicator_label.setText(f"{message}")

    def _stop_decomposition(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self._reset_ui_state()

    def _on_decomposition_error(self, err_msg):
        QMessageBox.critical(self, "Error", f"Decomposition Failed:\n{err_msg}")
        self._reset_ui_state()

    def _on_source_found(self, source, timestamps, iteration, silhouette):
        self._plot_source_realtime(source, timestamps, iteration, silhouette)
        QApplication.processEvents()

    def _plot_source_realtime(self, source, timestamps, iteration, silhouette):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS['background'])
        
        source_np = source.detach().cpu().numpy() if torch.is_tensor(source) else source
        timestamps_np = timestamps.detach().cpu().numpy() if torch.is_tensor(timestamps) else timestamps
        
        if len(timestamps_np) > 0:
            idx = timestamps_np.astype(int)
            idx = idx[idx < len(source_np)] 
            y_values = source_np[idx]
        else:
            idx = []
            y_values = []
        
        ax.plot(source_np, color='#2b6cb0', linewidth=1.2, alpha=0.9) 
        
        if len(idx) > 0:
            ax.plot(idx, y_values, 'o', color='#ed8936', markersize=4, alpha=0.9)

        ax.set_title(
            f"Iteration {iteration} | Silhouette: {silhouette:.3f} | {len(timestamps_np)} spikes",
            color=COLORS['foreground'], fontsize=12, weight='bold'
        )
        ax.set_xlabel("Sample", color=COLORS['foreground'])
        ax.set_ylabel("Amplitude", color=COLORS['foreground'])
        ax.tick_params(colors=COLORS['foreground'])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['foreground'])
        ax.spines['bottom'].set_color(COLORS['foreground'])
        
        self.canvas.draw()

    def _reset_ui_state(self):
        self._awaiting_time_confirmation = False
        self.start_btn.setText("Start Decomposition")
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.time_sel_widget.setVisible(False)