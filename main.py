import os
import sys
import time
from functools import partial
from itertools import count, cycle

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QPixmap
from PyQt5.QtNetwork import QUdpSocket
from PyQt5.QtWidgets import (QAction, QApplication, QComboBox, QHBoxLayout,
                             QInputDialog, QLabel, QMainWindow, QMenu,
                             QMessageBox, QPushButton, QSlider, QSplitter,
                             QStyle, QToolBar, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)


class MainData:
    categories = {
        'main': {
            'headers': [
                'MD', 'curr_27V', 'u_36V_C', 'u_36V_A', 'u_36V_B',
                'u15V_p_AP', 'u15V_m_AP', 'u27V_del', 'alfa', 'u_5V',
                'EA', 'EH', 'current', 'signal_D', 'Unn',
                'Una', 'D_analog', 'gamma', 'epsilon', 'psi',
                'ARU', 'E_H_ap', 'E_g', 'E_v', 'E_A_ap',
                'u_12V', 'u_12V_gnd', 'u_12V_m_018A', 'u_12V_m_018A_gnd', 'u_48V',
                'u_48V_gnd', 'u_8V', 'u_8V_gnd', 'u_6V_m_0075A', 'u_6V_m_gnd',
                'u_12V_0075A', 'u_12V_0075A_gnd', 'u_6V', 'u_6V_gnd', 'u_6V_m_028A',
                'u_6V_m_028A_gnd', 'zad_izc'
            ],
            'coef': [
                0.01, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                0.00244, 0.00244, 1, 0.00488, 1,
                1, 0.00488, 0.00244, 0.00244, 0.00244,
                1, 0.1, 1, 1, 0.1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 0.01,
            ],
            'types': [
                np.uint16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.int16, np.int16, np.int16, np.int16,
                np.int16, np.uint16
            ],
            'visible': True
        },
        'vid_data': {
            'headers': [
                'vid_data'
            ],
            'visible': False
        },
        'arinc': {
            'headers': [
                'ARINC_081', 'ARINC_082', 'ARINC_083', 'ARINC_084',
                'ARINC_085', 'ARINC_086', 'ARINC_087', 'ARINC_088',
                'ARINC_089'
            ],
            'visible': True
        },
        'bit_data': {
            'headers': [
                'send_ARINC', 'u27_p', 'u27_ground', 'u27_A', 'u27_A1', 'u36B_m', 'u36A_m', 'u15_m',
                'u15v_p', 'u36C_m', 'u15V_bk', 'off_vob', 'off_V', 'off_ASU', 'block_VP', 'off_CU',
                'vkl_rrch', 'PR_27v', 'block_AB', 'bridge27V', 'VPG_27V', 'komm_ASD', 'block_DP', 'sinhro',
                'RIP', 'D5', 'DVA1', 'DVA2', 'DVA3', 'DVA4', 'kom_vn', 'kontr_toka_rzp',
                'kontr_zahv_apch', 'kontr_vn', 'EhV', 'AV', 'PR_U505', 'Zg_27V', 'Kom_No', 'VK',
                'komm_PP', 'kom_mem_ASD', 'ASP', 'Tg_RAZI', 'MD_k', 'Tg_ZHO', 'ZH_ZH', 'Si_k',
                'PPH', 'Sh_P2', 'izp_k', 'izr_k', 'strob_RZ', 'zona_1', 'zona_2', 'rpo',
                'AR', 'kom_rg_rv', 'sz', 'kom_ASD_k', 'Tg_ZH_Zh', 'kom_vp', 'null_1', 'null_2',
                'kontrol_27V_m_pit', 'kontrol_27V_m', 'kontrol_27V_p_pit', 'kontrol_27V_p',
                'kontrol_27V_p_A0', 'kontrol_27V_p_A1', 'kontrol_27V_p_A2', 'kontrol_27V_p_A3'
            ],
            'visible': True
        },
        'time_src': {
            'headers': [
                'time_src'
            ],
            'visible': True
        }
    }

    def __init__(self):
        for category in self.categories.values():
            for name in category['headers']:
                setattr(self, name, [])
        self.counter = count()

    def __iter__(self):
        for category in self.categories:
            for name in category['headers']:
                yield name, getattr(self, name)

    def clear_data(self):
        self.__init__()

    def add_data(self, name, data):
        value = getattr(self, name)
        value.extend(data)
        # setattr(self, name, value)

    def cut_data(self):
        for category in self.categories.values():
            for name in category['headers']:
                value = getattr(self, name)
                setattr(self, name, value[-51_000:])
        self.counter = count()

    def get_object(self, name):
        return getattr(self, name)

    # def get_time(self, name):
    #     return getattr(self, 'time_src')

    def add_byte_data(self, data):
        dt = np.dtype([
            ('pack_header', np.uint8, (1, )),
            ('vid_data', np.uint8, 1024),
            ('null_bytes', np.uint8, 2),
            ('arinc_data', np.uint16, 9),
            ('bit_data', np.uint8, 9),
            ('null_bytes2', np.uint8, (1, )),
            ('data_main', np.int16, 42),
            ('null_bytes3', np.uint8, 56),
            ('time_src', np.uint32, (1, )),
            ('null_bytes4', np.uint8, 33)
        ])
        dt = dt.newbyteorder('>')
        res = np.frombuffer(data, dtype=dt, count=-1)
        self.unpack_data(res)

    def unpack_data(self, res):
        main_headers = self.categories['main']['headers']
        main_coef = self.categories['main']['coef']
        main_types = self.categories['main']['types']
        for index, (name, coef, type) in enumerate(zip(main_headers, main_coef, main_types)):
            self.add_data(name, res['data_main'][:, index].astype(type) * coef)
        self.add_data('time_src', res['time_src'][:, 0] * 0.002)
        # self.add_data('time_src', [next(counter) / 50_000])
        for index, col in enumerate(self.categories['arinc']['headers']):
            self.add_data(col, res['arinc_data'][:, index])

        columns_bits = [
            [
                'send_ARINC', 'u27_p', 'u27_ground', 'u27_A', 'u27_A1', 'u36B_m', 'u36A_m', 'u15_m'
            ],
            [
                'u15v_p', 'u36C_m', 'u15V_bk', 'off_vob', 'off_V', 'off_ASU', 'block_VP', 'off_CU'
            ],
            [
                'vkl_rrch', 'PR_27v', 'block_AB', 'bridge27V', 'VPG_27V', 'komm_ASD', 'block_DP', 'sinhro'
            ],
            [
                'RIP', 'D5', 'DVA1', 'DVA2', 'DVA3', 'DVA4', 'kom_vn', 'kontr_toka_rzp'
            ],
            [
                'kontr_zahv_apch', 'kontr_vn', 'EhV', 'AV', 'PR_U505', 'Zg_27V', 'Kom_No', 'VK'
            ],
            [
                'komm_PP', 'kom_mem_ASD', 'ASP', 'Tg_RAZI', 'MD_k', 'Tg_ZHO', 'ZH_ZH', 'Si_k'
            ],
            [
                'PPH', 'Sh_P2', 'izp_k', 'izr_k', 'strob_RZ', 'zona_1', 'zona_2', 'rpo'
            ],
            [
                'AR', 'kom_rg_rv', 'sz', 'kom_ASD_k', 'Tg_ZH_Zh', 'kom_vp', 'null_1', 'null_2'
            ],
            [
                'kontrol_27V_m_pit', 'kontrol_27V_m', 'kontrol_27V_p_pit', 'kontrol_27V_p', 'kontrol_27V_p_A0', 'kontrol_27V_p_A1', 'kontrol_27V_p_A2', 'kontrol_27V_p_A3'
            ]
        ]

        for i, val in enumerate(columns_bits):
            bit_data = self.unpack_bits(val, res['bit_data'][:, i])
            for key, value in bit_data.items():
                self.add_data(key, np.array(value))

        self.add_data('vid_data', res['vid_data'])

        counter = next(self.counter)
        if counter > 5_000:
            self.cut_data()

    @staticmethod
    def unpack_bits(columns, data):
        result = {}
        for i, name in enumerate(columns):
            result[name] = ((data & (1 << i)) >> i).astype(np.bool_)
        return result


class UpdateGrapicsThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        self.update_signal.emit()


class UpdateDataThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        self.update_signal.emit()


class IndicatorLabel(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.set_red()

    def set_red(self):
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.GlobalColor.red)
        self.setPixmap(pixmap)

    def set_green(self):
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.GlobalColor.green)
        self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.last_update = 0
        self.resolution = 10000
        self.data = MainData()
        self.graph_widgets = {}
        self.graph_vid_widget = None
        self.process_started = False
        self.socket = QUdpSocket()
        self.received_packets = 0
        self.packet_for_update = 50
        self.indicator_timer = QTimer(self)
        self.indicator_timer.timeout.connect(self.indicator_update)
        self.update_graphs_threads = UpdateGrapicsThread()
        self.update_graphs_threads.update_signal.connect(
            self.update_all_graphics
        )
        self.cache = b''
        self.update_data_threads = UpdateDataThread()
        self.update_data_threads.update_signal.connect(self.update_data)
        self.settings = QSettings('settings.ini', QSettings.IniFormat)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("VID GRAPH UPD v.2024.03.13")
        self.setGeometry(0, 0, 1350, 768)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        self.main_layout = QHBoxLayout(main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.left_widget = LeftMenuTree(self)

        self.right_widget = QWidget()
        self.right_widget.setStyleSheet(
            'QWidget {background-color: rgb(0, 0, 0)}')
        right_widget_layout = QVBoxLayout(self.right_widget)
        right_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.right_graph_layout = QVBoxLayout()
        right_widget_layout.addLayout(self.right_graph_layout, 3)
        self.right_graph_layout.setContentsMargins(0, 0, 0, 0)
        self.right_graph_layout.setSpacing(0)

        self.right_vid_layout = QVBoxLayout()
        right_widget_layout.addLayout(self.right_vid_layout, 1)
        self.right_vid_layout.setContentsMargins(0, 0, 0, 0)
        self.right_vid_layout.setSpacing(0)

        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_widget)

        self.set_action()
        self.set_toolbar()
        self.showMaximized()

    def create_action(self, text, icon, slot, checkable=False):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(icon)
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def set_action(self):
        self.start_process_action = self.create_action(
            'Запуск процесса', QIcon('play.png'), self.start_process)

    def set_toolbar(self):
        pos = Qt.ToolBarArea.TopToolBarArea
        toolbar = QToolBar()
        toolbar.addAction(self.start_process_action)

        toolbar.addSeparator()
        # self.diss_combobox = QComboBox()
        # self.diss_combobox.addItems(
        #     ['1', '2'])
        # self.diss_combobox.activated.connect(self.set_data)
        # self.diss_combobox.setFixedWidth(150)
        # toolbar.addWidget(self.diss_combobox)
        # toolbar.addSeparator()

        toolbar.addWidget(QLabel(' Масштаб:'))
        self.slider_resolution = QSlider(Qt.Orientation.Horizontal)
        self.slider_resolution.setFixedSize(100, 40)
        self.slider_resolution.setMaximum(50000)
        self.slider_resolution.setMinimum(1000)
        self.slider_resolution.setTickInterval(1000)
        self.slider_resolution.setValue(self.resolution)
        self.slider_resolution.valueChanged.connect(
            self.slider_resolution_handler)
        toolbar.addWidget(self.slider_resolution)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(' Скорость обновления графиков:'))
        self.slider_update = QSlider(Qt.Orientation.Horizontal)
        self.slider_update.setFixedSize(100, 40)
        self.slider_update.setMaximum(495)
        self.slider_update.setMinimum(300)
        self.slider_update.setTickInterval(10)
        self.slider_update.setValue(500 - self.packet_for_update)
        self.slider_update.valueChanged.connect(self.slider_update_handler)
        toolbar.addWidget(self.slider_update)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(' Пакетов получено: '))
        self.received_packets_label = QLabel(str(self.received_packets))
        toolbar.addWidget(self.received_packets_label)

        toolbar.addSeparator()
        self.indicator_label = IndicatorLabel()
        toolbar.addWidget(self.indicator_label)

        toolbar.addSeparator()
        button_create_vid_graph = QPushButton(
            'Показать видеосигнал')
        button_create_vid_graph.clicked.connect(self.create_vid_graph)
        toolbar.addWidget(button_create_vid_graph)

        toolbar.addSeparator()
        button_create_graphs_window = QPushButton(
            'Построить несколько графиков')
        button_create_graphs_window.clicked.connect(self.create_graph_window)
        toolbar.addWidget(button_create_graphs_window)

        toolbar.addSeparator()

        button_clear_graph = QPushButton('Очистить графики')
        button_clear_graph.clicked.connect(self.clear_graphs)
        toolbar.addWidget(button_clear_graph)

        self.view_menu = QMenu(self)
        self.view_menu.setTitle('Пресеты графиков')
        toolbar.addAction(self.view_menu.menuAction())
        self.update_view_menu()

        self.addToolBar(pos, toolbar)

    def clear_graphs(self):
        self.data.clear_data()
        self.update_all_graphics()

    def update_view_menu(self):
        self.view_menu.clear()

        save_current_action = QAction('Сохранить текущее отображение', self)
        save_current_action.triggered.connect(self.save_view)
        self.view_menu.addAction(save_current_action)

        for name, values in self.settings.value('view_settings', {}).items():
            sub_menu = QMenu(self.view_menu)
            self.view_menu.addMenu(sub_menu)
            sub_menu.setTitle(name)

            activate_action = QAction('Восстановить', self)
            activate_action.triggered.connect(
                partial(self.restore_view, values))
            sub_menu.addAction(activate_action)

            delete_action = QAction('Удалить отображение', self)
            delete_action.triggered.connect(
                partial(self.delete_view_from_settings, name))
            sub_menu.addAction(delete_action)

    def save_view(self):
        names_graph = [a for a in self.graph_widgets.keys()]
        if not names_graph:
            return

        text, ok_pressed = QInputDialog.getText(
            self, 'Введите имя', 'Имя: '
        )
        if not ok_pressed or not text:
            return

        current_values = self.settings.value('view_settings', {})
        current_values[text] = names_graph
        self.settings.setValue('view_settings', current_values)
        self.update_view_menu()

    def restore_view(self, list_names):
        for name in list(self.graph_widgets):
            self.delete_graph_window(name)
        for name in list_names:
            self.create_graph_window(name)

    def delete_view_from_settings(self, name):
        current_settings = self.settings.value('view_settings', {})
        if name in current_settings:
            current_settings.pop(name)
            self.settings.setValue('view_settings', current_settings)
            self.update_view_menu()

    def slider_resolution_handler(self):
        self.resolution = self.slider_resolution.value()
        self.update_graphs_threads.start()

    def slider_update_handler(self):
        self.packet_for_update = 500 - self.slider_update.value()

    def start_process(self):
        if self.process_started:
            self.stop_process()
            return

        self.process_started = True
        self.received_packets_for_update = 0

        self.start_process_action.setIcon(QIcon('stop.png'))
        self.indicator_timer.start(500)
        self.socket.bind(2015)
        self.socket.readyRead.connect(self.read_data)

    def stop_process(self):
        self.process_started = False
        self.start_process_action.setIcon(QIcon('play.png'))
        self.socket.close()
        self.received_packets = 0
        self.received_packets_label.setText(str(self.received_packets))
        self.received_packets_for_update = 0
        self.indicator_label.set_red()
        self.indicator_timer.stop()

    def read_data(self):
        while self.socket.hasPendingDatagrams():
            self.received_packets += 1
            self.received_packets_label.setText(f'{self.received_packets}')

            data, * _ = self.socket.readDatagram(1274)
            self.cache += data

            self.update_data_threads.start()

            if self.received_packets_for_update >= self.packet_for_update:
                self.update_graphs_threads.start()
                self.received_packets_for_update = 0

            self.received_packets_for_update += 1

            self.last_update = time.time_ns()

    def update_data(self):
        if self.cache:
            self.data.add_byte_data(self.cache)
            self.cache = b''

    def track_graph(self) -> None:
        widgets = list(self.graph_widgets.values())
        if not widgets:
            return
        main_child = widgets[-1]

        for widget in widgets[:-1]:
            widget.setXLink(main_child)
            widget.getAxis('bottom').setStyle(showValues=False)

    def create_graph_window(self, graph_names=False):
        if not graph_names:
            graph_names = self.left_widget.get_checked_element()
            graph_names = tuple(graph_names)
        if not graph_names:
            return

        if len(self.graph_widgets) > 9:
            QMessageBox.warning(
                self,
                'Предупреждение',
                'Достигнут лимит отображения графиков, закройте ненужные'
            )
            return

        if graph_names in self.graph_widgets:
            self.delete_graph_window(graph_names)

        graph_widget = GraphWidget(graph_names, self)
        self.graph_widgets[graph_names] = graph_widget
        self.right_graph_layout.addWidget(graph_widget)
        self.track_graph()

    def create_vid_graph(self):
        if self.graph_vid_widget is not None:
            self.graph_vid_widget.close()
            self.graph_vid_widget = None
        self.graph_vid_widget = VidGraph(self)
        self.right_vid_layout.addWidget(self.graph_vid_widget)

    def delete_graph_window(self, column_name):
        if column_name not in self.graph_widgets:
            return
        self.graph_widgets[column_name].close()
        del self.graph_widgets[column_name]
        self.track_graph()

    def update_all_graphics(self):
        for widget in self.graph_widgets.values():
            widget.update_data()
        if self.graph_vid_widget is not None:
            self.graph_vid_widget.update_data()

    def indicator_update(self):
        now = time.time_ns()
        if now - self.last_update >= 1000000000:
            self.indicator_label.set_red()
        else:
            self.indicator_label.set_green()

    def clear_window(self):
        if self.process_started:
            self.stop_process()

        self.left_widget.update_checkbox()

        for name in list(self.graph_widgets.keys()):
            self.delete_graph_window(name)

    def closeEvent(self, ev):
        if self.process_started:
            self.stop_process()
        super().closeEvent(ev)


class LeftMenuTree(QTreeWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setColumnCount(1)
        self.update_checkbox()
        self.itemDoubleClicked.connect(self.item_double_click_handle)

    def update_checkbox(self):
        self.clear()
        self.widgets = []
        self.setHeaderHidden(True)
        self.setFixedWidth(200)
        for category, column_data in self.main_window.data.categories.items():
            if not column_data['visible']:
                continue
            tree_category = QTreeWidgetItem(self)
            tree_category.setText(0, category)
            tree_category.setExpanded(True)

            for name in column_data['headers']:
                column_widget = QTreeWidgetItem(tree_category)
                column_widget.setText(0, name)
                column_widget.setCheckState(0, Qt.CheckState.Unchecked)
                self.widgets.append(column_widget)

    def item_double_click_handle(self, item: QTreeWidgetItem):
        if not item in self.widgets:
            return
        self.main_window.create_graph_window((item.text(0),))

    def get_checked_element(self):
        return [
            item.text(0) for item in self.widgets if item.checkState(0) == Qt.CheckState.Checked
        ]


class GraphWidget(pg.PlotWidget):
    colors = cycle([
        'red', 'green', 'blue', 'cyan',
        'purple', 'white', 'orange',
        'yellow', 'fuchsia', 'olive',
        'lime', 'aqua', 'maroon', 'teal'
    ])

    def __init__(self, graph_names, main_window):
        super().__init__()
        self.graph_names = graph_names
        self.main_window = main_window
        self.resolution = 0
        self.curves = {}
        self.getAxis('left').setWidth(50)

        close_action = QAction('Закрыть (Средняя клавиша мышки)')
        close_action.triggered.connect(
            lambda: self.main_window.delete_graph_window(self.graph_names))
        self.scene().contextMenu.append(close_action)

        self.create_graphs()

    def create_graphs(self):
        self.showGrid(x=True, y=True)
        self.apply_theme('black')
        # self.setMenuEnabled(False)
        self.setClipToView(True)
        self.setDownsampling(auto=True, mode='peak')

        for name in self.graph_names:
            color = next(self.colors)
            pen = pg.mkPen(color=color, width=2)
            oy = self.main_window.data.get_object(
                name)[-self.main_window.resolution:]
            ox = np.arange(len(oy)) * 0.002
            curve = pg.PlotDataItem(ox, oy, name=name, pen=pen, connect='all')
            self.addItem(curve)
            self.curves[name] = curve

        # self.proxy = pg.SignalProxy(
        #     self.scene().sigMouseMoved,
        #     rateLimit=30,
        #     slot=self.mouse_moved
        # )

        # self.scene().sigMouseClicked.connect(self.mouse_click_event)
        self.scene().sigMouseMoved.connect(self.mouse_moved)

    def mouse_moved(self, ev):
        if self.sceneBoundingRect().contains(ev):
            mousePoint = self.getPlotItem().vb.mapSceneToView(ev)
            self.pos_x = float(mousePoint.x())
            self.pos_y = float(mousePoint.y())
            self.setToolTip(
                # x: <b>{self.pos_x:.2f}</b>,<br> y:
                f'<b>{self.pos_y:.3f}</b>'
            )

    def clear_other_display_text(self):
        widgets = self.main_window.graph_widgets.values()
        for widget in widgets:
            if widget is not self:
                widget.display_text.setText('')

    # def mouse_moved(self, e) -> None:
    #     '''
    #     Метод высплывающей подсказки по координатам при перемещении мыши.
    #     '''
    #     pos = e[0]
    #     if self.sceneBoundingRect().contains(pos):
    #         mousePoint = self.getPlotItem().vb.mapSceneToView(pos)
    #         x = round(float(mousePoint.x()))
    #         y = round(float(mousePoint.y()), 1)
    #         times = ''
    #         for name in self.curves:
    #             data = self.main_window.data.get_time(name)
    #             if data is None or len(data) < 1:
    #                 continue
    #             data = data[-self.resolution:]
    #             if x < 0 or x >= len(data):
    #                 continue
    #             times += f'<br>Время {name}: <b>{data[x]}</b>'

    #         self.setToolTip(
    #             f'Номер пакета: <b>{round(x)}</b><br> Значение: <b>{round(y, 1)}</b>{times}'
    #         )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.main_window.delete_graph_window(self.graph_names)
        return super().mousePressEvent(ev)

    def apply_theme(self, color):
        self.setBackground(color)
        legend_color = 'black' if color == 'white' else 'white'
        pen = pg.mkPen(legend_color, width=1.0)
        for axis in ['bottom', 'left']:
            axis_obj = self.getAxis(axis)
            axis_obj.setPen(pen)
            axis_obj.setTextPen(pen)
        self.addLegend(
            pen=legend_color,
            labelTextColor=legend_color,
            offset=(0, 0)
        )

    def update_data(self):
        if self.resolution != self.main_window.resolution:
            self.resolution = self.main_window.resolution
            self.setXRange(0, self.resolution * 0.0021)
        for name, curve in self.curves.items():
            data = self.main_window.data.get_object(name)
            oy = data[-self.resolution:]
            ox = np.arange(len(oy)) * 0.002
            curve.setData(ox, oy)

    # def mouse_click_event(self, ev):
    #     if ev.button() == Qt.MouseButton.RightButton:
    #         ev.accept()
    #         self.context_menu(ev)
    #         return
    #     return super().mouse_click_event(ev)

    # def context_menu(self, ev):
    #     menu = QMenu()
    #     close_action = QAction('Закрыть (Средняя клавиша мышки)')
    #     close_action.triggered.connect(
    #         lambda: self.main_window.delete_graph_window(self.graph_names))
    #     menu.addAction(close_action)
    #     menu.exec(ev.screenPos().toPoint())


class VidGraph(pg.PlotWidget):
    def __init__(self, main_window, pos=-1):
        super().__init__()
        self.main_window = main_window
        self.curve = None
        self.pos = pos
        self.getAxis('left').setWidth(50)
        self.create_graphs()

    def create_graphs(self):
        self.showGrid(x=True, y=True)
        self.apply_theme('black')
        self.setMenuEnabled(False)
        self.setClipToView(True)
        self.setDownsampling(auto=True, mode='peak')
        self.scene().sigMouseClicked.connect(self.mouse_click_event)

        pen = pg.mkPen(width=2)
        data = self.main_window.data.get_object(
            'vid_data')

        oy = data[self.pos] if len(data) else []
        ox = list(range(len(oy)))
        curve = pg.PlotDataItem(ox, oy, name='vid_data',
                                pen=pen, connect='all')
        self.addItem(curve)
        self.curve = curve

    def mouse_click_event(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.context_menu(ev)
            ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.close()
        return super().mousePressEvent(ev)

    def apply_theme(self, color):
        self.setBackground(color)
        legend_color = 'black' if color == 'white' else 'white'
        pen = pg.mkPen(legend_color, width=1.0)
        for axis in ['bottom', 'left']:
            axis_obj = self.getAxis(axis)
            axis_obj.setPen(pen)
            axis_obj.setTextPen(pen)
        self.addLegend(
            pen=legend_color,
            labelTextColor=legend_color,
            offset=(0, 0)
        )

    def update_data(self):
        data = self.main_window.data.get_object('vid_data')
        oy = data[self.pos] if len(data) else []
        ox = np.arange(len(oy))
        self.curve.setData(ox, oy)

    def context_menu(self, ev):
        menu = QMenu()
        close_action = QAction('Закрыть (Средняя клавиша мышки)')
        close_action.triggered.connect(self.close)
        menu.addAction(close_action)
        menu.exec(ev.screenPos().toPoint())

    def closeEvent(self, ev):
        self.main_window.graph_vid_widget = None
        super().closeEvent(ev)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 10, 10))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(142, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    Stylesheet = '''
        QTreeView {
            show-decoration-selected: 1;
            outline: 0;
            background-color: rgb(10, 10, 10);
            color: white;
        }
        QTreeView::indicator {
            border: 1px solid gray;
        }
        QTreeView::indicator:checked {
            background-color: white;
        }
    '''

    app.setStyleSheet(Stylesheet)
    window = MainWindow(app)
    window.show()
    app.exec()
