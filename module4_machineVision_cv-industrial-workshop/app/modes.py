# วิธีตรวจทั้งหมดของสถานีตรวจชิ้นงาน  แต่ละฟังก์ชันตรงกับ notebook ใน Workshop
# ทุกฟังก์ชันรับ (ภาพ BGR, ค่าตั้ง dict) คืน dict: verdict, reason, stat, views=[(ภาพ, คำอธิบาย)]
import re
from datetime import date

import cv2
import numpy as np

BS = chr(92)


def _box(img, c, color, thick):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)


def _thick(img):
    return max(2, img.shape[1] // 250)


# ---------- Workshop 1: พื้นฐานภาพ ----------
def mode_basics(img, s):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    view = s['basics_view']
    if view == 'ภาพเทา':
        out, cap = gray, 'Gray = 0.299R + 0.587G + 0.114B'
    elif view == 'ชั้นสี B, G, R':
        b, g, r = cv2.split(img)
        out, cap = np.hstack([b, g, r]), 'ชั้น B | ชั้น G | ชั้น R  (สว่าง = มีสีนั้นมาก)'
    elif view == 'ชั้น H, S, V':
        h, sat, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        out, cap = np.hstack([(h.astype(np.float32) * 255 / 179).astype(np.uint8), sat, v]), 'H สีอะไร | S สดแค่ไหน | V สว่างแค่ไหน'
    elif view == 'Threshold':
        _, out = cv2.threshold(gray, s['th_T'], 255, cv2.THRESH_BINARY_INV if s['th_inv'] else cv2.THRESH_BINARY)
        cap = 'Binary Image  T = ' + str(s['th_T'])
    elif view == 'Canny หาขอบ':
        out, cap = cv2.Canny(gray, s['canny_lo'], s['canny_hi']), 'Canny (' + str(s['canny_lo']) + ', ' + str(s['canny_hi']) + ')'
    else:
        out, cap = img.copy(), 'ภาพต้นฉบับ'
    h, w = img.shape[:2]
    x = min(s['px_x'], w - 1); y = min(s['px_y'], h - 1)
    b, g, r = img[y, x]
    hh, ss, vv = cv2.cvtColor(img[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0]
    mark = img.copy()
    cv2.drawMarker(mark, (x, y), (0, 255, 255), cv2.MARKER_CROSS, max(20, w // 30), _thick(img))
    info = f'พิกเซล ({x}, {y})  BGR = ({b}, {g}, {r})  HSV = ({hh}, {ss}, {vv})'
    return dict(verdict='INFO', reason=info, stat=f'{w} x {h} x {img.shape[2]}',
                views=[(out, cap), (mark, 'ตำแหน่งพิกเซลที่เลือก')])


# ---------- Workshop 2 / 2B / 5: นับตามสี ----------
def _color(hsv, lo, hi, s_min, v_min):
    m = cv2.inRange(hsv, np.array([lo, s_min, v_min]), np.array([hi, 255, 255]))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return m, cs


def mode_color(img, s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo, hi = s['h_range']
    mask, cs = _color(hsv, lo, hi, s['s_min'], s['v_min'])
    out = img.copy(); t = _thick(img)
    big = [c for c in cs if cv2.contourArea(c) > s['min_area']]
    chips = [c for c in cs if s['min_area'] * 0.05 < cv2.contourArea(c) <= s['min_area']]
    foreign = []
    if s['chk_foreign']:
        _, cs_o = _color(hsv, 12, 22, s['s_min'], s['v_min'])
        foreign = [c for c in cs_o if cv2.contourArea(c) > s['min_area']]
    for c in big:
        _box(out, c, (40, 180, 40), t)
    for c in (chips if s['chk_chip'] else []) + foreign:
        _box(out, c, (0, 0, 255), t)
    n = len(big)
    if n < s['expect']:
        reason = 'MISSING  ขาด ' + str(s['expect'] - n) + ' ชิ้น'
    elif n > s['expect']:
        reason = 'EXTRA  เกิน ' + str(n - s['expect']) + ' ชิ้น'
    elif s['chk_chip'] and chips:
        reason = 'CHIP  มีเศษชิ้นแตก ' + str(len(chips))
    elif foreign:
        reason = 'FOREIGN  มีสิ่งแปลกปลอม ' + str(len(foreign))
    else:
        reason = 'ครบถ้วน'
    return dict(verdict='OK' if reason == 'ครบถ้วน' else 'NG', reason=reason,
                stat=str(n) + ' / ' + str(s['expect']),
                views=[(out, 'ผลการตรวจ (เขียว = นับ  แดง = ผิดปกติ)'), (mask, 'Mask สีที่เลือก')])


# ---------- Workshop 6 ข้อ 1 / 10: เทียบภาพอ้างอิง ----------
def align_ecc(img, ref):
    g0 = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    g1 = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    h, w = ref.shape[:2]
    try:
        score, warp = cv2.findTransformECC(g0, g1, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5), None, 5)
    except cv2.error:
        return img, np.full((h, w), 255, np.uint8), 0.0
    fl = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    out = cv2.warpAffine(img, warp, (w, h), flags=fl, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), warp, (w, h), flags=fl, borderValue=0)
    return out, cv2.erode(valid, np.ones((15, 15), np.uint8)), score


def mode_reference(img, s):
    ref = s.get('ref_img')
    if ref is None:
        return dict(verdict='INFO', reason='ยังไม่มีภาพอ้างอิง  กดตั้งภาพอ้างอิงในแท็บปรับเงื่อนไข', stat='-',
                    views=[(img, 'ภาพปัจจุบัน')])
    if img.shape[:2] != ref.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
    valid = np.full(ref.shape[:2], 255, np.uint8); score = 1.0
    if s['ref_align']:
        img, valid, score = align_ecc(img, ref)
    g1 = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g0 = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(g1, g0)
    _, m = cv2.threshold(diff, s['ref_th'], 255, cv2.THRESH_BINARY)
    m = cv2.bitwise_and(m, valid)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bad = [c for c in cs if cv2.contourArea(c) > s['ref_min_area']]
    out = img.copy(); t = _thick(img)
    for c in bad:
        _box(out, c, (0, 0, 255), t)
    reason = 'ตรงกับภาพอ้างอิง' if not bad else 'DIFF  พบจุดที่ต่าง ' + str(len(bad)) + ' จุด'
    if s['ref_align'] and score < 0.6:
        reason += '  (จัดตำแหน่งไม่ดี ' + f'{score:.2f}' + ')'
    return dict(verdict='OK' if not bad else 'NG', reason=reason, stat=str(len(bad)) + ' จุด',
                views=[(out, 'ผลการตรวจ (แดง = ต่างจากภาพอ้างอิง)'), (diff * 4, 'ความต่างจากภาพอ้างอิง (ขยาย 4 เท่า)'),
                       (ref, 'ภาพอ้างอิง')])


# ---------- Workshop 4 / 6 ข้อ 2: ตรวจฉลากด้วย OCR ----------
LOT_RE = 'LOT' + BS + 's*(' + BS + 'd{6})'
EXP_RE = 'EXP' + BS + 's*(' + BS + 'd{4})-(' + BS + 'd{2})-(' + BS + 'd{2})'


def mode_label(img, s):
    try:
        import pytesseract
    except ImportError:
        return dict(verdict='INFO', reason='ยังไม่ได้ติดตั้ง pytesseract / Tesseract', stat='-', views=[(img, 'ภาพ')])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(bw, config='--psm 6')
    except Exception as e:
        return dict(verdict='INFO', reason='OCR ไม่ทำงาน: ' + str(e)[:80], stat='-', views=[(img, 'ภาพ')])
    flat = ' '.join(text.split())
    reasons = []
    lot = re.search(LOT_RE, flat, re.IGNORECASE)
    exp = re.search(EXP_RE, flat, re.IGNORECASE)
    if s['lbl_need_lot'] and not lot:
        reasons.append('NO_LOT')
    if s['lbl_need_exp']:
        if not exp:
            reasons.append('NO_EXP')
        else:
            try:
                d = (date(int(exp.group(1)), int(exp.group(2)), int(exp.group(3))) - s['lbl_today']).days
                if d < s['lbl_min_days']:
                    reasons.append('SHORT_LIFE เหลือ ' + str(d) + ' วัน')
            except ValueError:
                reasons.append('BAD_DATE')
    want = s['lbl_product'].strip().upper()
    if want:
        import difflib
        up = flat.upper(); L = len(want)
        sim = max((difflib.SequenceMatcher(None, want, up[i:i + L]).ratio() for i in range(max(1, len(up) - L + 1))), default=0)
        if sim < 0.85:
            reasons.append('WRONG_PRODUCT')
    stat = ('LOT ' + lot.group(1) if lot else 'ไม่พบ LOT') + ('  EXP ' + exp.group(0)[3:].strip() if exp else '')
    return dict(verdict='OK' if not reasons else 'NG', reason='ครบถ้วน' if not reasons else ', '.join(reasons),
                stat=stat, views=[(img, 'อ่านได้: ' + (flat[:120] or '(ไม่พบตัวอักษร)')), (bw, 'หลัง Threshold (Otsu)')])


# ---------- Workshop 6 ข้อ 3: นับและแยกชนิดชิ้นงานโลหะ ----------
def mode_parts(img, s):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(bw); cv2.drawContours(filled, cs, -1, 255, -1)
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(bw))
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return dict(verdict='NG', reason='ไม่พบชิ้นงาน', stat='0', views=[(img, 'ภาพ'), (bw, 'Threshold')])
    _, peaks = cv2.threshold(dist, s['peak_frac'] * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    sure_bg = cv2.dilate(filled, np.ones((9, 9), np.uint8))
    n_m, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[(sure_bg == 255) & (peaks == 0)] = 0
    markers = cv2.watershed(img, markers)
    out = img.copy(); out[markers == -1] = (0, 0, 255)
    nw = nn = 0
    for i in range(2, n_m + 1):
        reg = markers == i
        if reg.sum() < s['parts_min_area']:
            continue
        hole = int((holes[reg] > 0).sum())
        washer = hole > s['hole_min']
        nw += washer; nn += not washer
        ys, xs = np.where(reg)
        cv2.putText(out, 'W' if washer else 'N', (int(xs.mean()) - 8, int(ys.mean()) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if washer else (0, 200, 255), 2)
    ok = nw == s['exp_washer'] and nn == s['exp_nut']
    reason = 'ครบถ้วน' if ok else ('แหวน ' + str(nw) + '/' + str(s['exp_washer']) + '  น็อต ' + str(nn) + '/' + str(s['exp_nut']))
    return dict(verdict='OK' if ok else 'NG', reason=reason, stat='W ' + str(nw) + '  N ' + str(nn),
                views=[(out, 'W = แหวน  N = น็อต  เส้นแดง = รอยแยก Watershed'), (bw, 'หลัง Threshold (Otsu)')])


# ---------- Workshop 6 ข้อ 4: วัดขนาด ----------
def mode_measure(img, s):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flag = cv2.THRESH_BINARY_INV if s['meas_dark'] else cv2.THRESH_BINARY
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, flag + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted([c for c in cs if cv2.contourArea(c) > 500], key=lambda c: cv2.boundingRect(c)[0])
    if len(cs) < 2:
        return dict(verdict='NG', reason='ต้องมีวัตถุสอบเทียบ (ซ้ายสุด) และชิ้นงานอย่างน้อย 1 ชิ้น', stat='-',
                    views=[(img, 'ภาพ'), (bw, 'Threshold')])
    (cx, cy), r = cv2.minEnclosingCircle(cs[0])
    ppm = 2 * r / s['meas_ref_mm']
    out = img.copy(); t = _thick(img)
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 200, 0), t)
    rect = cv2.minAreaRect(cs[1])
    W, H = sorted([rect[1][0] / ppm, rect[1][1] / ppm], reverse=True)
    cv2.drawContours(out, [cv2.boxPoints(rect).astype(int)], 0, (0, 0, 255), t)
    cv2.putText(out, f'{W:.1f} x {H:.1f} mm', (int(rect[0][0]) - 80, int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(out, f'{W:.1f} x {H:.1f} mm', (int(rect[0][0]) - 80, int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    ok = abs(W - s['meas_w']) <= s['meas_tol'] and abs(H - s['meas_h']) <= s['meas_tol']
    ew, eh, et = s['meas_w'], s['meas_h'], s['meas_tol']
    reason = 'ขนาดอยู่ในเกณฑ์' if ok else f'SIZE  {W:.1f} x {H:.1f} mm  (ต้องการ {ew:.1f} x {eh:.1f} ± {et})'
    return dict(verdict='OK' if ok else 'NG', reason=reason, stat=f'{W:.1f} x {H:.1f} mm',
                views=[(out, f'สอบเทียบ {ppm:.2f} px/mm  (1 px = {1 / ppm:.2f} mm)'), (bw, 'หลัง Threshold')])


# ---------- Workshop 8: แผนที่โซน (ROI map) ----------
RACK_ZONES = ('bay1,drive,50,80,80,170' + chr(10) + 'bay2,drive,138,80,80,170' + chr(10) +
              'bay3,drive,226,80,80,170' + chr(10) + 'bay4,drive,314,80,80,170' + chr(10) +
              'bay5,drive,402,80,80,170' + chr(10) + 'bay6,drive,490,80,80,170' + chr(10) +
              'bay7,drive,578,80,80,170' + chr(10) + 'bay8,drive,666,80,80,170' + chr(10) +
              'led_power,led,48,28,24,24' + chr(10) + 'led_status,led,88,28,24,24' + chr(10) +
              'usb1,port,640,28,36,24' + chr(10) + 'usb2,port,690,28,36,24' + chr(10) + 'label,label,300,22,260,36')


def parse_zones(text):
    zones = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) == 6:
            try:
                zones.append((p[0], p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5])))
            except ValueError:
                pass
    return zones


def _zone_ok(kind, roi, ref_roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if kind == 'drive':
        return (gray > 90).mean() > 0.6
    if kind == 'led':
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (40, 80, 80), (80, 255, 255)).mean() / 255 > 0.3
    if kind == 'port':
        return (gray > 150).mean() > 0.25
    if kind == 'label':
        return (gray > 200).mean() > 0.5
    if kind == 'ref' and ref_roi is not None:
        g0 = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY).astype(float)
        return abs(gray.mean() - g0.mean()) < 25 and abs(gray.std() - g0.std()) < 25
    return True


def mode_zones(img, s):
    zones = parse_zones(s['zones_text'])
    ref = s.get('ref_img')
    out = img.copy(); bad = []; t = _thick(img)
    for name, kind, x, y, w, h in zones:
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        rr = ref[y:y + h, x:x + w] if ref is not None and ref.shape == img.shape else None
        ok = _zone_ok(kind, roi, rr)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0) if ok else (0, 0, 255), t)
        if not ok:
            bad.append(name)
            cv2.putText(out, name, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return dict(verdict='OK' if not bad else 'NG', reason='ครบทุกโซน' if not bad else 'ส่วนที่ผิด: ' + ', '.join(bad),
                stat=str(len(zones) - len(bad)) + ' / ' + str(len(zones)) + ' โซน',
                views=[(out, 'เขียว = ผ่าน  แดง = ผิด')])


# ---------- Workshop 7: AI จำแนกภาพ ----------
from vision_camera import make_hog  # noqa: E402
_HOG = make_hog()


def ai_features(img):
    small = cv2.resize(img, (64, 48)); gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 24)).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    col = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    return np.concatenate([_HOG.compute(gray).flatten(), (thumb / (thumb.mean() + 1e-6)).flatten(), 10 * col / (col.sum() + 1e-6)])


def make_tray(kind, rng):
    img = np.full((240, 320, 3), (70, 72, 75), np.uint8)
    dx, dy = rng.integers(-10, 11, 2)
    centers = [(60 + dx, 60 + dy), (160 + dx, 60 + dy), (260 + dx, 60 + dy),
               (60 + dx, 170 + dy), (160 + dx, 170 + dy), (260 + dx, 170 + dy)]
    skip = rng.integers(0, 6) if kind == 'missing' else -1
    for i, (x, y) in enumerate(centers):
        if i == skip:
            continue
        cv2.circle(img, (int(x), int(y)), 26, (190, 190, 195), -1)
        cv2.circle(img, (int(x), int(y)), 8, (80, 82, 85), -1)
    if kind == 'foreign':
        fx, fy = int(rng.integers(30, 290)), int(rng.integers(30, 210))
        f = rng.integers(0, 3)
        if f == 0:
            cv2.rectangle(img, (fx, fy), (fx + int(rng.integers(40, 70)), fy + 12), (30, 30, 35), -1)
        elif f == 1:
            cv2.circle(img, (fx, fy), int(rng.integers(14, 22)), (60, 200, 230), -1)
        else:
            pts = np.array([[fx - 30, fy], [fx, fy - 12], [fx + 30, fy + 10], [fx + 60, fy - 5]], np.int32)
            cv2.polylines(img, [pts], False, (20, 20, 20), 3)
    light = rng.uniform(0.6, 1.3)
    return np.clip(img.astype(np.float32) * light + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def demo_dataset(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in ('ok', 'missing', 'foreign'):
        for _ in range(n):
            X.append(ai_features(make_tray(label, rng))); y.append(label)
    return X, y


def train_ai(X, y):
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), SVC(C=10)).fit(np.array(X), np.array(y))


def mode_ai(img, s):
    model = s.get('ai_model')
    if model is None:
        return dict(verdict='INFO', reason='ยังไม่มีโมเดล  เก็บภาพตัวอย่างหรือใช้ชุดตัวอย่าง แล้วกดสอน AI ในแท็บปรับเงื่อนไข',
                    stat='-', views=[(img, 'ภาพปัจจุบัน')])
    label = str(model.predict([ai_features(img)])[0])
    ok_labels = [x.strip() for x in s['ai_ok_labels'].split(',')]
    return dict(verdict='OK' if label in ok_labels else 'NG', reason='AI ทายว่า: ' + label,
                stat=label, views=[(img, 'AI ทายว่า ' + label)])


# ---------- Workshop 5: สายพานจำลอง ----------
def conveyor_scene(trays, t_ms, speed, gap=200, fov_w=640):
    # ถาดวิ่งจากซ้ายไปขวาด้วยความเร็ว speed (พิกเซล/มิลลิวินาที)  คืนภาพสายพานทั้งเส้น และตำแหน่งกลางภาพ (เซนเซอร์)
    th, tw = trays[0].shape[:2]
    W = fov_w * 3
    scene = np.full((th, W, 3), (55, 57, 60), np.uint8)
    off = int(t_ms * speed) % 80
    for x in range(off - 80, W, 80):
        cv2.line(scene, (x, 0), (x, th), (68, 70, 74), 6)
    pitch = tw + gap
    travel = t_ms * speed
    k_hi = int(travel // pitch)
    for k in range(max(0, k_hi - W // pitch - 2), k_hi + 1):
        x0 = int(travel - k * pitch) - tw
        if x0 >= W or x0 + tw <= 0:
            continue
        a, b = max(0, x0), min(W, x0 + tw)
        scene[:, a:b] = trays[k % len(trays)][:, a - x0:b - x0]
    return scene, W // 2


def trigger_time(k, speed, tw, W, gap=200):
    # เวลาที่ถาดที่ k มีจุดกลางผ่านเซนเซอร์ (กลางสายพาน)
    return (W // 2 + tw / 2 + k * (tw + gap)) / speed


def capture(scene, center, fov_w, blur_px):
    crop = scene[:, center - fov_w // 2:center + fov_w // 2].copy()
    L = int(round(blur_px))
    if L > 1:
        k = np.full((1, L), 1.0 / L, np.float32)
        crop = cv2.filter2D(crop, -1, k)
    return crop


MODES = {
    'basics': ('พื้นฐานภาพ (Workshop 1)', mode_basics),
    'color': ('นับตามสี (Workshop 2, 2B, 5)', mode_color),
    'reference': ('เทียบภาพอ้างอิง (Workshop 6, 10)', mode_reference),
    'label': ('ตรวจฉลาก OCR (Workshop 4, 6)', mode_label),
    'parts': ('นับและแยกชนิดโลหะ (Workshop 6)', mode_parts),
    'measure': ('วัดขนาด (Workshop 6)', mode_measure),
    'zones': ('แผนที่โซน ROI (Workshop 8)', mode_zones),
    'ai': ('AI จำแนกภาพ (Workshop 7)', mode_ai),
    'conveyor': ('สายพานจำลอง (Workshop 5)', mode_color),
}
