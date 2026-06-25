# ============================================
# pages/home.py - صفحه اصلی با Fashion DNA و Outfit Generator
# ============================================
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from utils.body_detection import detect_body_shape
from utils.skin_analysis import extract_skin_color, classify_skin_tone, get_manual_skin_info
from utils.dataset import get_final_recommendations
from utils.ui_components import display_product_with_image, show_metric_cards, show_color_palette
from utils.pdf_generator import generate_pdf_report
from utils.fashion_engine import generate_fashion_dna, explain_body_shape, generate_complete_outfit


# ============================================
# استایل‌های CSS
# ============================================
def inject_css():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px 35px;
            border-radius: 20px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        .main-header h1 { font-size: 2rem; margin: 0; }
        .main-header p { opacity: 0.9; margin: 5px 0 0 0; }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-top: 4px solid #667eea;
            transition: transform 0.3s;
        }
        .metric-card:hover { transform: translateY(-5px); }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label { color: #888; font-size: 0.85rem; margin-top: 5px; }

        .main-card {
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 5px 25px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }

        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 15px 0;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        }
        .result-card h2 { color: white; margin-top: 0; }
        .result-card td { color: white; padding: 6px 5px; }
        .result-card strong { color: #FFD700; }

        .do-box { 
            background: #D4EDDA; 
            border-right: 4px solid #28A745; 
            padding: 10px; 
            border-radius: 8px; 
            margin: 6px 0;
            color: #155724;
        }
        .dont-box { 
            background: #F8D7DA; 
            border-right: 4px solid #DC3545; 
            padding: 10px; 
            border-radius: 8px; 
            margin: 6px 0;
            color: #721c24;
        }

        .product-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.06);
            margin: 10px 0;
            transition: transform 0.3s;
        }
        .product-card:hover { transform: translateY(-5px); }

        /* Fashion DNA Card */
        .dna-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 25px;
            border-radius: 15px;
            color: #00FF88;
            text-align: center;
            font-family: 'Courier New', monospace;
            margin: 15px 0;
            border: 2px solid #00FF8844;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================
# تابع اصلی
# ============================================
def show_home(df, image_folder):
    inject_css()

    st.markdown(
        '<div class="main-header"><h1>👗 Smart Style Assistant</h1><p>هوش مصنوعی، استایل شما را متحول می‌کند</p></div>',
        unsafe_allow_html=True)

    show_metric_cards(df, st.session_state.analysis_count)
    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("### 📸 تحلیل استایل")

        gender = st.radio("👤 جنسیت", ['female', 'male'], format_func=lambda x: '👩 خانم' if x == 'female' else '👨 آقا',
                          horizontal=True)
        occasion = st.selectbox("🎯 موقعیت", ['casual', 'formal', 'party'],
                                format_func=lambda x: {'casual': '👕 روزمره', 'formal': '👔 رسمی', 'party': '🎉 مهمانی'}[
                                    x])

        skin_method = st.radio("🎨 رنگ پوست", ['auto', 'manual'],
                               format_func=lambda x: '🤖 خودکار' if x == 'auto' else '✋ دستی', horizontal=True)
        if skin_method == 'manual':
            if 'manual_skin' not in st.session_state:
                st.session_state.manual_skin = None
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            for i, (code, label) in enumerate(
                    [('fair', '🥛 سفید'), ('medium', '🌾 گندمی'), ('olive', '🫒 سبزه'), ('dark', '🍫 تیره')]):
                with [col_s1, col_s2, col_s3, col_s4][i]:
                    if st.button(label, key=f"sk_{code}", use_container_width=True):
                        st.session_state.manual_skin = code

        uploaded_file = st.file_uploader("📸 عکس تمام‌قد", type=["jpg", "jpeg", "png"])
        analyze_btn = st.button("✨ شروع تحلیل", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if uploaded_file and analyze_btn:
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            shape_data, annotated, error_msg = detect_body_shape(image_cv, gender)

            if error_msg:
                st.error(error_msg)
                st.stop()

            if shape_data is not None:
                if skin_method == 'manual' and st.session_state.get('manual_skin'):
                    skin_info = get_manual_skin_info(st.session_state.manual_skin)
                    detection_method = '✋ انتخاب دستی'
                else:
                    skin_rgb = extract_skin_color(image_cv)
                    skin_info = classify_skin_tone(skin_rgb)
                    detection_method = '🤖 تشخیص خودکار'

                st.session_state.analysis_count += 1
                st.session_state.history.append({
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'shape': shape_data['farsi'],
                    'confidence': f"{shape_data['confidence']:.1f}%",
                    'skin': skin_info['skin_tone'],
                    'undertone': skin_info.get('undertone', 'نامشخص'),
                    'gender': gender
                })

                annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                st.image(annotated_pil, caption="🔴 شانه‌ها | 🟢 کمر | 🟣 باسن", use_container_width=True)

                st.markdown(f'''
                <div class="result-card">
                    <h2>📊 نتیجه تحلیل #{st.session_state.analysis_count}</h2>
                    <table>
                        <tr><td><strong>🎯 نوع اندام:</strong></td><td>{shape_data['farsi']}</td></tr>
                        <tr><td><strong>📊 Confidence:</strong></td><td>{shape_data['confidence']:.1f}%</td></tr>
                        <tr><td><strong>📐 SHR:</strong></td><td>{shape_data['shr']:.2f}</td></tr>
                        <tr><td><strong>📐 WHR:</strong></td><td>{shape_data['whr']:.2f}</td></tr>
                        <tr><td><strong>🎨 رنگ پوست:</strong></td><td>{skin_info['skin_tone']}</td></tr>
                        <tr><td><strong>✨ ته‌رنگ:</strong></td><td>{skin_info.get('undertone', 'نامشخص')}</td></tr>
                    </table>
                </div>
                ''', unsafe_allow_html=True)

                # ============================================
                # 🧬 Fashion DNA
                # ============================================
                st.markdown("---")
                st.markdown("### 🧬 Fashion DNA Code")

                style_personality = "Unknown"
                if 'style_profile' in st.session_state:
                    primary = st.session_state.style_profile.get('primary', '')
                    secondary = st.session_state.style_profile.get('secondary', '')
                    if secondary and st.session_state.style_profile['percentages'].get(secondary, 0) > 20:
                        style_personality = f"{primary} {secondary}"
                    else:
                        style_personality = primary

                dna = generate_fashion_dna(
                    body_shape=shape_data['english'],
                    skin_tone=skin_info['skin_code'],
                    style_personality=style_personality,
                    gender=gender
                )

                st.markdown(f"""
                <div class="dna-card">
                    <p style="font-size: 0.9rem; color: #aaa; margin: 0;">YOUR FASHION DNA CODE</p>
                    <h1 style="font-size: 3rem; letter-spacing: 8px; margin: 10px 0;">{dna['code']}</h1>
                    <p style="font-size: 0.8rem; color: #888; margin: 0;">Body: {dna['body_shape']} | Skin: {dna['skin_tone']} | Style: {dna['style_personality']}</p>
                </div>
                """, unsafe_allow_html=True)

                # ============================================
                # 🔬 Explainable AI
                # ============================================
                st.markdown("---")
                st.markdown("### 🔬 Explainable AI - چرا این تشخیص؟")

                explanations = explain_body_shape(shape_data)
                for exp in explanations:
                    with st.expander(f"{exp['icon']} {exp['title']}: {exp['detail']}"):
                        st.markdown(f"**{exp['explanation']}**")

                # ============================================
                # 👗 Outfit Generator
                # ============================================
                st.markdown("---")
                st.markdown("### 👗 استایل پیشنهادی امروز شما")

                outfit = generate_complete_outfit(
                    body_shape=shape_data['english'],
                    gender=gender,
                    occasion=occasion,
                    skin_tone=skin_info['skin_code'],
                    df=df
                )

                if outfit:
                    cols = st.columns(len(outfit))
                    category_icons = {'top': '👕', 'bottom': '👖', 'outerwear': '🧥', 'shoes': '👟', 'accessories': '⌚'}
                    category_names = {'top': 'بالاتنه', 'bottom': 'پایین‌تنه', 'outerwear': 'کت/پالتو', 'shoes': 'کفش',
                                      'accessories': 'اکسسوری'}

                    for i, (cat, item) in enumerate(outfit.items()):
                        if i < len(cols):
                            with cols[i]:
                                st.markdown(f"""
                                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                                    <div style="font-size: 2rem;">{category_icons.get(cat, '👗')}</div>
                                    <p style="font-weight: bold; margin: 5px 0; color: #666;">{category_names.get(cat, cat)}</p>
                                    <p style="font-size: 0.9rem; margin: 5px 0;">{item['name']}</p>
                                    <p style="color: #888; font-size: 0.8rem;">🎨 {item['color']}</p>
                                </div>
                                """, unsafe_allow_html=True)

                # ============================================
                # پالت رنگی
                # ============================================
                st.markdown("---")
                st.markdown("### 🎨 پالت رنگی پیشنهادی")
                show_color_palette(skin_info['colors'])

                # ============================================
                # محصولات از دیتاست
                # ============================================
                st.markdown("---")
                st.markdown("### 🛍️ محصولات پیشنهادی از دیتاست")

                if df is not None:
                    recommendations = get_final_recommendations(df, shape_data['english'], gender, occasion,
                                                                skin_info['colors'])
                    if not recommendations.empty:
                        st.success(f"✅ {len(recommendations)} محصول یافت شد!")
                        for _, product in recommendations.head(6).iterrows():
                            st.markdown('<div class="product-card">', unsafe_allow_html=True)
                            display_product_with_image(product, image_folder)
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("محصولی یافت نشد.")

                # ============================================
                # PDF + امتیاز
                # ============================================
                col_pdf, col_rate = st.columns(2)
                with col_pdf:
                    pdf_data = generate_pdf_report(shape_data, skin_info, recommendations, gender, occasion)
                    st.download_button("📥 دانلود PDF", data=pdf_data,
                                       file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       mime="application/pdf", use_container_width=True)
                with col_rate:
                    st.slider("⭐ امتیاز دهید", 1, 5, 4, key=f"rate_{st.session_state.analysis_count}")
            else:
                st.error("❌ بدن تشخیص داده نشد.")
        elif uploaded_file:
            st.image(uploaded_file, caption="👤 عکس آپلود شده", use_container_width=True)
            st.info("👈 دکمه تحلیل را بزنید.")
        else:
            st.markdown("<div style='text-align:center;padding:60px 20px;color:#999;'><h2>📸 منتظر عکس شما</h2></div>",
                        unsafe_allow_html=True)