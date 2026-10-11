# สถานีตรวจชิ้นงาน (Web app) รองรับทุก Workshop  รันด้วย:  python3 -m streamlit run app/inspection_app.py
import os
import sys
import threading
import time
from datetime import date, datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
sys.path.insert(0, HERE)
from vision_camera import BaslerSource, WebcamSource, set_param, simulate_real, img_path  # noqa: E402
import modes as M  # noqa: E402

st.set_page_config(page_title='สถานีตรวจชิ้นงาน', page_icon='📷', layout='wide')
Q = chr(34)

# ---------------- สูตรงาน: 1 สูตร = 1 Workshop ----------------
R = {}
def recipe(name, desc, mode, **kw):
    R[name] = dict(desc=desc, mode=mode, **kw)

recipe('W1 พื้นฐานภาพ: พิกเซล สี Threshold Canny', 'ดูภาพเป็นตัวเลข แยกชั้นสี ลอง Threshold และหาขอบ', 'basics',
       source='ภาพตัวอย่าง', image='shapes.png')
recipe('W2 นับวัตถุสีแดง', 'นับชิ้นแดงใหญ่ ต้องได้ 5 ชิ้น', 'color', source='ภาพตัวอย่าง', image='shapes.png')
recipe('W2 นับวัตถุสีน้ำเงิน', 'เปลี่ยนช่วงสีเป็นน้ำเงิน H 95–125', 'color', source='ภาพตัวอย่าง', image='shapes.png',
       h_range=(95, 125), expect=3)
recipe('W2B ตรวจถาดสายการผลิต (กล้องจำลอง)', 'ครบ 5 ชิ้น + เศษแตก + สิ่งแปลกปลอมสีส้ม  ลองถาด 01–06', 'color',
       source='กล้อง Basler จำลอง', image='line/tray_01.png', chk_chip=True, chk_foreign=True)
recipe('W4 ตรวจฉลาก OCR', 'อ่าน LOT และ EXP แล้วตรวจกฎ', 'label', source='ภาพตัวอย่าง', image='ocr_sample.png')
recipe('W5 จำลองกะการผลิต: แสงน้อย', 'แสงเหลือ 20%  ลองแก้ด้วย Exposure หรือ V ต่ำสุด', 'color', source='ภาพตัวอย่าง',
       image='factory/part_001.png', chk_chip=True, chk_foreign=True, sim=True, light=0.2)
recipe('W5 จำลองกะการผลิต: สายพานเร็ว', 'ภาพเบลอ  ลองลด Exposure แล้วเพิ่ม Gain', 'color', source='ภาพตัวอย่าง',
       image='factory/part_001.png', chk_chip=True, chk_foreign=True, sim=True, speed=2.0, exposure=10.0)
recipe('W5 สายพานจำลอง: ความเร็ว + Trigger', 'ถาดวิ่งผ่านกล้อง เซนเซอร์สั่งถ่ายเมื่อถาดถึงกลางภาพ  กดเริ่มสายพาน แล้วลองเร่งความเร็ว เพิ่ม Exposure หรือหน่วง Trigger', 'conveyor',
       source='ภาพตัวอย่าง', image='line/tray_01.png', chk_chip=True, chk_foreign=True)
recipe('W6 สิ่งแปลกปลอม: เทียบภาพอ้างอิง', 'ภาพอ้างอิง = ถาดดี  ลองถาดที่มีสกรู เส้นผม ชิ้นหาย', 'reference',
       source='ภาพตัวอย่าง', image='industrial/tray_02_screw.png', ref_file='industrial/tray_reference.png')
recipe('W6 ตรวจฉลากสินค้า', 'ชื่อสินค้าต้องเป็น PRODUCT A  มี LOT และ EXP', 'label', source='ภาพตัวอย่าง',
       image='industrial/label_01_good.png', lbl_product='PRODUCT A')
recipe('W6 นับแหวนและน็อต', 'ไม่ใช้สี  แยกชิ้นที่ติดกันด้วย Watershed', 'parts', source='ภาพตัวอย่าง',
       image='industrial/parts_01.png')
recipe('W6 วัดขนาด', 'วัตถุซ้ายสุด = เหรียญ 20 มม. ใช้สอบเทียบ', 'measure', source='ภาพตัวอย่าง',
       image='industrial/measure_01.png')
recipe('W7 AI จำแนกภาพ', 'แท็บปรับเงื่อนไข: ใช้ชุดตัวอย่าง → สอน AI แล้วลองภาพ ai/*  ลองภาพ industrial/tray_* ด้วย: AI ทายผิดเพราะภาพต่างจากชุดสอน', 'ai',
       source='ภาพตัวอย่าง', image='ai/foreign_01.png')
recipe('W8 Rack server: แผนที่โซน', 'ตรวจทีละส่วน: ไดรฟ์ 8 ช่อง ไฟสถานะ พอร์ต ฉลาก', 'zones', source='ภาพตัวอย่าง',
       image='rack/rack_03_led_fault.png')
recipe('W10 โครงการของฉัน (อัปโหลดรูปมือถือ)', 'อัปโหลดรูปชิ้นงานดีเป็นภาพอ้างอิง แล้วอัปโหลดรูปที่จะตรวจ', 'reference',
       source='อัปโหลดภาพ', ref_align=True, ref_th=25)
recipe('กล้องจริง: นับตามสี', 'ตั้งภาพใน pylon Viewer ก่อน  ภาพจริงใหญ่ พื้นที่ขั้นต่ำจึงสูง', 'color',
       source='กล้อง Basler จริง', s_min=80, v_min=60, min_area=15000, expect=1)

SOURCES = {'ภาพตัวอย่าง': 'file', 'อัปโหลดภาพ': 'upload', 'กล้อง Basler จำลอง': 'emu',
           'กล้อง Basler จริง': 'basler', 'Webcam': 'webcam'}
DEFAULTS = dict(
    mode='color', source='ภาพตัวอย่าง', image='shapes.png', sim=False, light=1.0, speed=0.0, exposure=5.0, gain=0.0,
    basics_view='ภาพต้นฉบับ', th_T=128, th_inv=False, canny_lo=50, canny_hi=150, px_x=100, px_y=100,
    h_range=(0, 10), s_min=50, v_min=50, min_area=1000, expect=5, chk_chip=False, chk_foreign=False,
    ref_file='', ref_align=False, ref_th=20, ref_min_area=120,
    lbl_need_lot=True, lbl_need_exp=True, lbl_min_days=30, lbl_product='', lbl_today=date.today(),
    peak_frac=0.5, hole_min=280, parts_min_area=300, exp_washer=7, exp_nut=5,
    meas_ref_mm=20.0, meas_dark=True, meas_w=25.5, meas_h=16.0, meas_tol=0.5,
    zones_text=M.RACK_ZONES, ai_ok_labels='ok', conv_speed=0.5, conv_delay=0, conv_proc=10)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k, v in (('conv_on', False), ('conv_t0', 0.0), ('conv_k', 0), ('conv_busy', -1e9), ('conv_log', []),
             ('conv_last', None)):
    st.session_state.setdefault(k, v)
for k, v in (('log', []), ('last', None), ('recipe', list(R)[0]), ('ref_img', None), ('ai_model', None),
             ('ai_X', []), ('ai_y', [])):
    st.session_state.setdefault(k, v)


def apply_recipe():
    r = R[st.session_state.recipe]
    for k, v in DEFAULTS.items():
        st.session_state[k] = r.get(k, v)
    st.session_state.last = None
    st.session_state.ref_img = cv2.imread(img_path(r['ref_file'])) if r.get('ref_file') else None


if st.session_state.recipe == list(R)[0] and st.session_state.mode != R[list(R)[0]]['mode']:
    apply_recipe()


# ---------------- กล้อง: เปิดค้างไว้ตัวเดียว ปิดตัวเก่าก่อนเปิดใหม่ ----------------
@st.cache_resource
def camera_holder():
    return {'key': None, 'cam': None, 'lock': threading.Lock()}


def close_camera():
    h = camera_holder()
    with h['lock']:
        if h['cam'] is not None:
            try:
                h['cam'].close()
            except Exception:
                pass
        h['key'], h['cam'] = None, None


def open_camera(kind, image, keep=False):
    h = camera_holder()
    key = (kind, image if kind == 'emu' else None, keep)
    if h['cam'] is not None and h['key'] == key:
        return h['cam']
    close_camera()
    with h['lock']:
        cam = WebcamSource() if kind == 'webcam' else BaslerSource(
            emulate=(kind == 'emu'), keep_settings=keep, image_path=img_path(image))
        cam.lock = threading.Lock(); cam.settings = None
        h['key'], h['cam'] = key, cam
    return cam


def sample_images():
    names = ['shapes.png', 'ocr_sample.png']
    for folder in ('line', 'factory', 'industrial', 'rack', 'ai'):
        d = img_path(folder)
        if os.path.isdir(d):
            names += [folder + '/' + f for f in sorted(os.listdir(d)) if f.endswith('.png')]
    return names


def small(img, width=960):
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hue_bar(lo, hi):
    h = np.tile(np.arange(180, dtype=np.uint8), (24, 1))
    bar = cv2.cvtColor(np.dstack([h, np.full_like(h, 200), np.full_like(h, 230)]), cv2.COLOR_HSV2RGB)
    sel = (h >= lo) & (h <= hi)
    bar[~sel] = (bar[~sel] * 0.25 + 190).astype(np.uint8)
    return cv2.resize(bar, (720, 26), interpolation=cv2.INTER_NEAREST)


def badge(verdict, text):
    color = {'OK': '#1E7E34', 'NG': '#C62828'}.get(verdict, '#3B4A5A')
    return ('<div style=' + Q + 'background:' + color + ';color:white;border-radius:14px;padding:10px 18px;text-align:center' + Q + '>'
            '<div style=' + Q + 'font-size:40px;font-weight:800;line-height:1.1' + Q + '>' + verdict + '</div>'
            '<div style=' + Q + 'font-size:17px;opacity:0.95' + Q + '>' + text + '</div></div>')


# ================= แถบซ้าย =================
with st.sidebar:
    st.markdown('### 1. เลือก Workshop')
    st.selectbox('สูตรงาน', list(R), key='recipe', on_change=apply_recipe, label_visibility='collapsed')
    st.caption(R[st.session_state.recipe]['desc'])
    st.button('↺ คืนค่าตามสูตร', on_click=apply_recipe, width='stretch')
    mode = st.session_state.mode
    st.caption('วิธีตรวจ: ' + M.MODES[mode][0])

    st.markdown('### 2. แหล่งภาพ')
    source_label = st.selectbox('ภาพมาจาก', list(SOURCES), key='source')
    source = SOURCES[source_label]
    is_camera = source in ('emu', 'basler', 'webcam')
    uploaded = None
    if source in ('file', 'emu'):
        imgs = sample_images()
        if st.session_state.image not in imgs:
            st.session_state.image = imgs[0]
        st.selectbox('ภาพชิ้นงาน', imgs, key='image')
    if source == 'upload':
        uploaded = st.file_uploader('เลือกไฟล์ภาพ (รูปมือถือได้)', type=['png', 'jpg', 'jpeg', 'bmp'])

    keep = False
    if is_camera or source == 'file':
        with st.expander('3. ตั้งค่ากล้อง', expanded=(source == 'basler')):
            if source == 'basler':
                keep = st.toggle('ใช้ค่าที่ตั้งใน pylon Viewer', value=True,
                                 help='ปิดตัวเลือกนี้ก่อนกด Trigger ถ้าจะปรับ Exposure ในแอป')
            st.slider('Exposure (มิลลิวินาที)', 0.5, 200.0, key='exposure', step=0.5, disabled=keep,
                      help='นานขึ้น = สว่างขึ้น แต่ชิ้นงานที่เคลื่อนที่จะเบลอ  กล้องจริงแนะนำ 20–100')
            st.slider('Gain', 0.0, 24.0, key='gain', step=1.0, disabled=keep,
                      help='สูงขึ้น = สว่างขึ้น แต่จุดรบกวนมากขึ้น')
            if source == 'basler' and st.button('White Balance อัตโนมัติ 1 ครั้ง', width='stretch'):
                try:
                    cam = open_camera('basler', st.session_state.image, keep)
                    with cam.lock:
                        set_param(cam.cam, ['BalanceWhiteAuto'], 'Once')
                    st.success('ปรับแล้ว ใช้กับภาพถัดไป')
                except Exception as e:
                    st.error('ปรับไม่ได้: ' + str(e)[:150])
            if is_camera and st.button('ปิดกล้อง (คืนให้โปรแกรมอื่น)', width='stretch'):
                close_camera(); st.success('ปิดกล้องแล้ว')

    if source in ('file', 'emu', 'upload'):
        with st.expander('4. จำลองสภาพโรงงาน', expanded=st.session_state.sim):
            st.toggle('เปิดการจำลอง', key='sim', help='ภาพไฟล์และกล้องจำลองไม่เปลี่ยนตาม Exposure/Gain ส่วนนี้จำลองผลแบบกล้องจริง')
            st.slider('แสงรอบไลน์', 0.1, 1.5, key='light', step=0.1, disabled=not st.session_state.sim)
            st.slider('ความเร็วสายพาน', 0.0, 4.0, key='speed', step=0.5, disabled=not st.session_state.sim)


def get_frame():
    if source == 'upload':
        if uploaded is None:
            return None
        return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if source == 'file':
        return cv2.imread(img_path(st.session_state.image))
    cam = open_camera(source, st.session_state.image, keep)
    try:
        with cam.lock:
            if isinstance(cam, BaslerSource) and not keep:
                want = (int(st.session_state.exposure * 1000), st.session_state.gain)
                if cam.settings != want:
                    set_param(cam.cam, ['ExposureTime', 'ExposureTimeAbs'], want[0])
                    set_param(cam.cam, ['Gain', 'GainRaw'], want[1])
                    cam.settings = want
            for _ in range(5):
                f = cam.read()
                if f is not None:
                    return f
    except Exception:
        close_camera()
        raise RuntimeError('การเชื่อมต่อกล้องหลุด กำลังเชื่อมต่อใหม่ ลองอีกครั้ง')
    raise RuntimeError('ภาพจากกล้องมาไม่ครบ ลองใหม่ หรือเช็กสาย LAN / PoE')


def prepare(frame):
    if st.session_state.sim and source != 'basler':
        return simulate_real(frame, st.session_state.exposure, st.session_state.gain,
                             st.session_state.light, st.session_state.speed)
    return frame


def settings():
    s = {k: st.session_state[k] for k in DEFAULTS}
    s['ref_img'] = st.session_state.ref_img
    s['ai_model'] = st.session_state.ai_model
    return s


# ================= หน้าหลัก =================
st.markdown('## 📷 สถานีตรวจชิ้นงาน')
top = st.columns([1.1, 1.0, 1.3, 2.6])
trigger = top[0].button('📷  Trigger', type='primary', width='stretch', help='ถ่าย 1 ภาพ ตรวจ และบันทึกผล')
live = top[1].toggle('ภาพสด', disabled=not is_camera, help='ตรวจต่อเนื่อง (เฉพาะแหล่งภาพที่เป็นกล้อง)')
stat_slot, badge_slot = top[2].empty(), top[3].empty()
tab_view, tab_tune, tab_log = st.tabs(['🔍 ผลตรวจ', '🎚️ ปรับเงื่อนไขตรวจ', '📋 บันทึกผล'])

# ---------- แท็บปรับเงื่อนไข: แสดงเฉพาะค่าของวิธีตรวจที่เลือก ----------
with tab_tune:
    st.caption('ปรับแล้วภาพล่าสุดถูกตรวจใหม่ทันที  ไม่ต้องกด Trigger ซ้ำ')
    c1, c2 = st.columns(2)
    if mode == 'basics':
        c1.radio('มุมมอง', ['ภาพต้นฉบับ', 'ภาพเทา', 'ชั้นสี B, G, R', 'ชั้น H, S, V', 'Threshold', 'Canny หาขอบ'], key='basics_view')
        c1.slider('Threshold T', 0, 255, key='th_T'); c1.checkbox('กลับขาวดำ (วัตถุเข้มบนพื้นสว่าง)', key='th_inv')
        c2.slider('Canny low', 0, 255, key='canny_lo'); c2.slider('Canny high', 0, 255, key='canny_hi')
        c2.number_input('พิกเซล x', 0, 5000, key='px_x'); c2.number_input('พิกเซล y', 0, 5000, key='px_y')
    elif mode == 'conveyor':
        with c1:
            st.slider('ความเร็วสายพาน (พิกเซล/มิลลิวินาที)', 0.1, 3.0, key='conv_speed', step=0.1,
                      help='ภาพเบลอ = ความเร็ว × Exposure  (ปรับ Exposure ที่แถบซ้าย ข้อ 3)')
            st.slider('หน่วง Trigger (มิลลิวินาที)', -150, 150, key='conv_delay', step=5,
                      help='เซนเซอร์ติดผิดตำแหน่ง หรือสัญญาณช้า ถาดจะไม่อยู่กลางภาพ')
            st.slider('เวลาประมวลผลต่อถาด (มิลลิวินาที)', 1, 1500, key='conv_proc', step=10,
                      help='ถ้าถาดถัดไปมาถึงก่อนประมวลผลเสร็จ = ตรวจไม่ทัน (พลาด)')
        with c2:
            st.slider('ช่วงสี H (Hue)', 0, 179, key='h_range')
            st.number_input('จำนวนที่ต้องมี', 0, 100, key='expect')
            st.checkbox('NG ถ้ามีเศษชิ้นแตก', key='chk_chip'); st.checkbox('NG ถ้ามีสิ่งแปลกปลอมสีส้ม', key='chk_foreign')
    elif mode == 'color':
        with c1:
            st.slider('ช่วงสี H (Hue)', 0, 179, key='h_range', help='แดง 0–10  ส้ม 12–22  เหลือง 22–35  เขียว 40–80  น้ำเงิน 95–125')
            st.image(hue_bar(*st.session_state.h_range), caption='สีที่สว่าง = ช่วงที่เลือก', width='stretch')
            st.slider('S ต่ำสุด (ความสด)', 0, 255, key='s_min'); st.slider('V ต่ำสุด (ความสว่าง)', 0, 255, key='v_min')
        with c2:
            st.number_input('พื้นที่ขั้นต่ำ (พิกเซล)', 0, 500000, key='min_area', step=100)
            st.number_input('จำนวนที่ต้องมี', 0, 100, key='expect')
            st.checkbox('NG ถ้ามีเศษชิ้นแตก', key='chk_chip'); st.checkbox('NG ถ้ามีสิ่งแปลกปลอมสีส้ม', key='chk_foreign')
    elif mode == 'reference':
        with c1:
            if st.button('📌 ตั้งภาพล่าสุดเป็นภาพอ้างอิง (ชิ้นงานดี)', width='stretch', disabled=st.session_state.last is None):
                st.session_state.ref_img = st.session_state.last.copy(); st.success('ตั้งภาพอ้างอิงแล้ว')
            ref_up = st.file_uploader('หรืออัปโหลดภาพอ้างอิง', type=['png', 'jpg', 'jpeg'], key='ref_upload')
            if ref_up is not None:
                st.session_state.ref_img = cv2.imdecode(np.frombuffer(ref_up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if st.session_state.ref_img is not None:
                st.image(small(st.session_state.ref_img, 480), caption='ภาพอ้างอิงปัจจุบัน', width='stretch')
        with c2:
            st.slider('ความต่างขั้นต่ำ (diff threshold)', 5, 100, key='ref_th', help='ต่ำ = ไวขึ้น แต่จับจุดรบกวนด้วย')
            st.number_input('พื้นที่ความต่างขั้นต่ำ (พิกเซล)', 0, 100000, key='ref_min_area', step=10)
            st.toggle('จัดตำแหน่งภาพก่อนเทียบ (รูปมือถือ / ชิ้นงานขยับ)', key='ref_align')
    elif mode == 'label':
        c1.text_input('ชื่อสินค้าที่ต้องมี (เว้นว่าง = ไม่ตรวจ)', key='lbl_product')
        c1.checkbox('ต้องมี LOT 6 หลัก', key='lbl_need_lot'); c1.checkbox('ต้องมี EXP', key='lbl_need_exp')
        c2.number_input('อายุคงเหลือขั้นต่ำ (วัน)', 0, 3650, key='lbl_min_days'); c2.date_input('วันที่ตรวจ', key='lbl_today')
    elif mode == 'parts':
        c1.number_input('แหวนที่ต้องมี', 0, 100, key='exp_washer'); c1.number_input('น็อตที่ต้องมี', 0, 100, key='exp_nut')
        c2.slider('ระดับแยกชิ้นที่ติดกัน (peak_frac)', 0.2, 0.9, key='peak_frac', step=0.05, help='ต่ำ = ชิ้นติดกันไม่แยก  สูง = ชิ้นเดียวถูกตัด')
        c2.number_input('ขนาดรูขั้นต่ำของแหวน (พิกเซล)', 0, 100000, key='hole_min', step=10)
        c2.number_input('พื้นที่ชิ้นงานขั้นต่ำ', 0, 100000, key='parts_min_area', step=50)
    elif mode == 'measure':
        c1.number_input('ขนาดจริงวัตถุสอบเทียบ (มม.)', 1.0, 500.0, key='meas_ref_mm', step=0.5)
        c1.checkbox('วัตถุเข้มบนพื้นสว่าง', key='meas_dark')
        c2.number_input('ความกว้างที่ต้องการ (มม.)', 0.0, 1000.0, key='meas_w', step=0.1)
        c2.number_input('ความสูงที่ต้องการ (มม.)', 0.0, 1000.0, key='meas_h', step=0.1)
        c2.number_input('ค่าคลาดเคลื่อนที่ยอมรับ ± (มม.)', 0.0, 100.0, key='meas_tol', step=0.1)
    elif mode == 'zones':
        c1.text_area('โซน: ชื่อ,ชนิด,x,y,w,h  (ชนิด: drive, led, port, label, ref)', key='zones_text', height=320)
        c2.markdown('**ชนิดโซน**  drive = ต้องมีชิ้นสว่าง  led = ต้องเป็นไฟเขียว  port = ต้องมีพอร์ต  label = ต้องมีฉลากขาว  ref = ต้องเหมือนภาพอ้างอิง')
        if c2.button('📌 ตั้งภาพล่าสุดเป็นภาพอ้างอิง (สำหรับชนิด ref)', disabled=st.session_state.last is None):
            st.session_state.ref_img = st.session_state.last.copy()
    elif mode == 'ai':
        with c1:
            st.markdown('**ชุดข้อมูลสอน AI**  ตอนนี้ ' + str(len(st.session_state.ai_y)) + ' ภาพ ' +
                        str(sorted(set(st.session_state.ai_y))))
            if st.button('ใช้ชุดตัวอย่างถาด 180 ภาพ (notebook 07)', width='stretch'):
                X, y = M.demo_dataset(60); st.session_state.ai_X, st.session_state.ai_y = list(X), list(y)
            lab = st.text_input('ชื่อประเภทของภาพล่าสุด', value='ok')
            if st.button('➕ เก็บภาพล่าสุดเป็นตัวอย่าง', width='stretch', disabled=st.session_state.last is None):
                st.session_state.ai_X.append(M.ai_features(st.session_state.last)); st.session_state.ai_y.append(lab.strip())
            if st.button('ล้างชุดข้อมูล', width='stretch'):
                st.session_state.ai_X, st.session_state.ai_y, st.session_state.ai_model = [], [], None
        with c2:
            st.text_input('ประเภทที่ถือว่า OK (คั่นด้วย ,)', key='ai_ok_labels')
            ys = st.session_state.ai_y
            ready = len(set(ys)) >= 2 and min(ys.count(c) for c in set(ys)) >= 3 if ys else False
            if st.button('🧠 สอน AI', type='primary', width='stretch', disabled=not ready,
                         help='ต้องมีอย่างน้อย 2 ประเภท ประเภทละ 3 ภาพ'):
                t0 = time.perf_counter()
                st.session_state.ai_model = M.train_ai(st.session_state.ai_X, st.session_state.ai_y)
                st.success('สอนเสร็จใน ' + f'{time.perf_counter() - t0:.2f}' + ' วินาที')
            st.caption('สถานะโมเดล: ' + ('พร้อมใช้' if st.session_state.ai_model is not None else 'ยังไม่ได้สอน'))

# ---------- ได้ภาพ ----------
new_shot = False
if trigger:
    try:
        with st.spinner('กำลังถ่ายภาพ...'):
            raw = get_frame()
        if raw is None:
            st.warning('ยังไม่ได้เลือกไฟล์ภาพทางซ้าย')
        else:
            st.session_state.last = prepare(raw); new_shot = True
    except Exception as e:
        st.error('ถ่ายภาพไม่สำเร็จ: ' + str(e)[:200])
elif st.session_state.last is None and source == 'file':
    st.session_state.last = prepare(cv2.imread(img_path(st.session_state.image)))


def render(frame, logging=False):
    t0 = time.perf_counter()
    res = M.MODES[mode][1](frame, settings())
    ms = (time.perf_counter() - t0) * 1000
    badge_slot.markdown(badge(res['verdict'], res['reason']), unsafe_allow_html=True)
    stat_slot.metric('ผล', res['stat'], help='เวลาตรวจ ' + str(round(ms)) + ' ms')
    views = res['views']
    if len(views) == 1:
        st.image(small(views[0][0], 1400), caption=views[0][1], width='stretch')
    else:
        a, b = st.columns([1.6, 1])
        a.image(small(views[0][0]), caption=views[0][1] + '   |   ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0])
                + '  ความสว่างเฉลี่ย ' + str(int(frame.mean())), width='stretch')
        for img, cap in views[1:]:
            b.image(small(img, 640), caption=cap, width='stretch')
    if logging:
        st.session_state.log.append({
            'เวลา': datetime.now().strftime('%H:%M:%S'), 'Workshop': st.session_state.recipe,
            'ภาพ': st.session_state.image if source in ('file', 'emu') else source_label,
            'ผล': res['verdict'], 'สาเหตุ': res['reason'], 'ค่า': res['stat'], 'เวลาตรวจ (ms)': round(ms, 1)})
    return res


@st.fragment(run_every=(1.0 if source == 'basler' else 0.3) if live else None)
def live_view():
    if not live:
        return
    try:
        f = prepare(get_frame()); st.session_state.last = f; render(f)
    except Exception as e:
        st.warning('ภาพนี้ไม่สำเร็จ: ' + str(e)[:150])


@st.cache_data
def conveyor_trays():
    names = sorted(f for f in os.listdir(img_path('line')) if f.endswith('.png'))
    return names, [cv2.imread(img_path('line/' + f)) for f in names]


@st.fragment(run_every=0.3 if st.session_state.get('conv_on') else None)
def conveyor_view():
    names, trays = conveyor_trays()
    sp = st.session_state.conv_speed
    exp = st.session_state.exposure
    th, tw = trays[0].shape[:2]
    W = 640 * 3
    pitch = tw + 200
    now = (time.time() - st.session_state.conv_t0) * 1000 if st.session_state.conv_on else 0
    s = settings()
    done = 0
    while st.session_state.conv_on and done < 5:
        k = st.session_state.conv_k
        t_k = M.trigger_time(k, sp, tw, W) + st.session_state.conv_delay
        if t_k > now:
            break
        name = names[k % len(names)]
        truth = M.mode_color(trays[k % len(trays)], s)['verdict']
        row = {'ถาดที่': k + 1, 'ภาพ': name, 'เฉลย': truth}
        if t_k < st.session_state.conv_busy:
            row.update({'ผลตรวจ': 'พลาด', 'สาเหตุ': 'ตรวจไม่ทัน (กำลังประมวลผลถาดก่อน)'})
        else:
            scene, c = M.conveyor_scene(trays, t_k, sp)
            img = M.capture(scene, c, 640, sp * exp)
            if st.session_state.sim:
                img = simulate_real(img, exp, st.session_state.gain, st.session_state.light, 0)
            res = M.mode_color(img, s)
            st.session_state.conv_busy = t_k + st.session_state.conv_proc
            st.session_state.conv_last = (img, res)
            row.update({'ผลตรวจ': res['verdict'], 'สาเหตุ': res['reason']})
        row['ถูกต้อง'] = 'ใช่' if row['ผลตรวจ'] == truth else 'ไม่'
        st.session_state.conv_log.append(row)
        st.session_state.conv_k += 1
        done += 1
    scene, c = M.conveyor_scene(trays, now, sp)
    view = scene.copy()
    cv2.rectangle(view, (c - 320, 2), (c + 320, th - 3), (0, 220, 255), 4)
    cv2.line(view, (c, 0), (c, th), (0, 0, 255), 3)
    st.image(small(view, 1400), caption='สายพาน  กรอบเหลือง = ภาพที่กล้องเห็น  เส้นแดง = เซนเซอร์สั่ง Trigger', width='stretch')
    log = st.session_state.conv_log
    per_min = sp * 60000 / pitch
    m = st.columns(5)
    m[0].metric('ถาด/นาที', f'{per_min:.0f}')
    m[1].metric('ภาพเบลอ (พิกเซล)', f'{sp * exp:.1f}')
    m[2].metric('ตรวจแล้ว', len(log))
    m[3].metric('ตัดสินถูก', sum(r['ถูกต้อง'] == 'ใช่' for r in log))
    m[4].metric('พลาด / ผิด', sum(r['ถูกต้อง'] == 'ไม่' for r in log))
    if st.session_state.conv_last is not None:
        img, res = st.session_state.conv_last
        badge_slot.markdown(badge(res['verdict'], res['reason']), unsafe_allow_html=True)
        stat_slot.metric('ผล', res['stat'])
        a, b = st.columns([1.3, 1])
        a.image(small(res['views'][0][0], 800), caption='ภาพล่าสุดที่กล้องถ่าย', width='stretch')
        if log:
            b.dataframe(pd.DataFrame(log).iloc[::-1].head(8), width='stretch', hide_index=True)


def conveyor_toggle():
    if st.session_state.conv_on:
        st.session_state.conv_on = False
    else:
        st.session_state.conv_on = True
        st.session_state.conv_t0 = time.time()
        st.session_state.conv_k, st.session_state.conv_busy = 0, -1e9
        st.session_state.conv_log, st.session_state.conv_last = [], None


with tab_view:
    if mode == 'conveyor':
        st.button('⏹ หยุดสายพาน' if st.session_state.conv_on else '▶ เริ่มสายพาน', on_click=conveyor_toggle,
                  type='primary', width='stretch')
        conveyor_view()
    elif live:
        live_view()
        if new_shot:
            r = render(st.session_state.last, logging=True); st.toast('บันทึกผล: ' + r['verdict'] + '  ' + r['reason'])
    elif st.session_state.last is not None:
        render(st.session_state.last, logging=new_shot)
    else:
        badge_slot.info('เลือก Workshop และแหล่งภาพทางซ้าย แล้วกด Trigger')

# ---------- แท็บบันทึกผล ----------
with tab_log:
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        ok = int((df['ผล'] == 'OK').sum()); ng = int((df['ผล'] == 'NG').sum())
        m = st.columns(4)
        m[0].metric('ตรวจทั้งหมด', len(df)); m[1].metric('OK', ok); m[2].metric('NG', ng)
        m[3].metric('Yield', (str(round(ok / (ok + ng) * 100)) + '%') if ok + ng else '-')
        ngs = df[df['ผล'] == 'NG']
        if len(ngs):
            st.markdown('**NG แยกตามสาเหตุ**')
            st.bar_chart(ngs['สาเหตุ'].str.split().str[0].value_counts())
        st.dataframe(df.iloc[::-1], width='stretch', hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button('⬇️ ดาวน์โหลด CSV', df.to_csv(index=False).encode('utf-8-sig'), 'inspection_log.csv',
                           'text/csv', width='stretch')
        if d2.button('ล้างบันทึก', width='stretch'):
            st.session_state.log = []; st.rerun()
    else:
        st.caption('ยังไม่มีการตรวจ  กด Trigger เพื่อบันทึกผล')
