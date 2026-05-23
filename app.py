import streamlit as st
import google.generativeai as genai
from supabase import create_client, Client
from audio_recorder_streamlit import audio_recorder
import json
import datetime
from datetime import timedelta
from gtts import gTTS
import io
from collections import defaultdict

# ==========================================
# 1. 초기 설정 및 모바일 UI 최적화
# ==========================================
st.set_page_config(page_title="AI Tutor - 멀티모델", page_icon="💬", layout="centered")

# 디자인 시스템 (기존 CSS 유지 및 말하기 가이드 스타일 추가)
st.markdown("""
<style>
    .stApp { max-width: 600px; margin: 0 auto; }
    .correction-box { background-color: #FEF3C7; color: #92400E; padding: 12px; border-radius: 10px; font-size: 14px; margin-bottom: 10px; border-left: 5px solid #F59E0B; }
    .pinned-word { border: 2px solid #EF4444; background-color: #FEF2F2; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .normal-word { border: 1px solid #E5E7EB; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .speaking-guide { background-color: #EFF6FF; padding: 15px; border-radius: 12px; border-left: 5px solid #3B82F6; font-size: 13px; color: #1E40AF; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# API 및 DB 초기화
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# 최신 고성능 모델 인스턴스 설정
main_chat_model = genai.GenerativeModel('gemini-3-flash-preview')
analyzer_model = genai.GenerativeModel('gemini-3.1-flash-lite', generation_config={"response_mime_type": "application/json"})
audio_coach_model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

# 세션 상태(Session State) 안정적 누적 초기화
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_word" not in st.session_state:
    st.session_state.quiz_word = None
if "quiz_options" not in st.session_state:
    st.session_state.quiz_options = []
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None
if "pronunciation_evals" not in st.session_state:
    st.session_state.pronunciation_evals = {}
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []  
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False  
if "quiz_type" not in st.session_state:
    st.session_state.quiz_type = "en_to_ko"  
if "quiz_type_pool" not in st.session_state:
    st.session_state.quiz_type_pool = []  
# 🌟 신규 추가: 말하기 모드 제어용 상태 변수
if "voice_stt_preview" not in st.session_state:
    st.session_state.voice_stt_preview = None  
if "autoplay_tts" not in st.session_state:
    st.session_state.autoplay_tts = None  

# ==========================================
# 2. 멀티 모델 분업 처리 엔진 (백엔드 함수)
# ==========================================
def run_voice_coach(target_expr, audio_bytes):
    """[모델 3] 천사 코치 모드 발음 평가 (Gemini 2.5 Flash)"""
    prompt = f"""
    당신은 세상에서 가장 다정하고 격려를 아끼지 않는 '천사' 원어민 발음 코치입니다.
    사용자가 외국어로 말하는 용기를 낸 것 자체를 아주 자랑스럽게 생각하세요.

    사용자가 읽으려고 시도한 목표 문장: "{target_expr}"
    (오디오 파일 첨부됨)

    [평가 가이드 - 샌드위치 피드백]
    1. 폭풍 칭찬: 오디오를 듣고 억양, 목소리 톤, 특정 발음 등 조금이라도 잘한 부분을 찾아 먼저 크게 칭찬하세요.
    2. 부드러운 교정: 여러 개가 어색하더라도 좌절하지 않도록 가장 중요한 딱 1~2개만 골라 아주 부드럽게 짚어주세요.
    3. 점수: 점수는 아주 후하게 주세요! (최소 65점 이상 부여)

    [출력 형식 - 반드시 JSON만 출력]
    {{
        "score": 65~100 사이의 점수,
        "feedback": "다정한 칭찬 + 부드러운 문제점 지적 + 따뜻한 격려가 합쳐진 문단",
        "correction_tip": "혀의 위치나 입모양 등 따라하기 쉬운 구체적이고 긍정적인 발음 꿀팁 1개"
    }}
    """
    audio_part = {"mime_type": "audio/wav", "data": audio_bytes}
    response = audio_coach_model.generate_content([prompt, audio_part])
    return json.loads(response.text)

def run_background_analysis(user_message, review_words):
    """[모델 2] 유연한 채팅형 문법 교정 및 단어 추출 (Gemini 3.1 Flash Lite)"""
    prompt = f"""
    당신은 센스 있고 세련된 원어민 회화 선생님입니다. 사용자의 문장을 분석하여 정해진 JSON 양식으로만 출력하세요. 인사말이나 부연 설명은 절대 하지 마세요.

    사용자 문장: "{user_message}"
    복습 목표 단어: {review_words}

    [분석 미션]
    1. 유연한 문법 교정: 소문자 'i' 사용, 쉼표 남용, 마침표 생략 등 모바일 채팅의 사소한 습관은 절대 지적하지 마세요!
       대신 한국식 영어(Konglish), 치명적인 시제 오류 등 '진짜 원어민처럼 말하기 위해 필요한 교정'만 제시하고, 고친 이유를 반드시 **한국어**로 친절하게 설명하세요. (자연스럽다면 null 처리)
    2. 사용자가 '복습 목표 단어'를 적절하게 사용했는지 검사하세요.
    3. 오늘 상황에 어울리는 유용한 실전 표현을 1개 추천하세요.

    [출력 형식 - 반드시 JSON만 출력]
    {{
        "correction": "올바른 교정 문장 (이유에 대한 한국어 설명) 형태로 작성. 완벽하다면 null",
        "review_check": {{"목표단어": true/false}},
        "new_learning": {{"type": "word" 또는 "expression", "item": "새로운 표현", "meaning": "뜻"}}
    }}
    """
    response = analyzer_model.generate_content(prompt)
    return json.loads(response.text)

# def run_main_chat(user_message, settings, review_words):
#     """[모델 1] 메인 대화 엔진 (Gemini 3 Flash)"""
#     system_prompt = f"""
#     당신은 {settings['target_language']} 회화 선생님입니다. 성별은 {settings['partner_gender']}이며, 다정하고 대화를 잘 이끌어냅니다.
#     현재 상황은 '{settings['situation']}' (Level: {settings['level']}) 입니다. 사용자의 레벨에 맞춰 너무 길지 않게 대답하세요.

#     [필수 미션]
#     1. 복습: 사용자가 이전에 어려워했던 단어 리스트 {review_words} 중 하나를 이번 대답에 자연스럽게 녹여서 사용해 보세요.
#     2. 대화가 끊기지 않도록 항상 마지막에는 가벼운 질문을 던져주세요.
#     3. 오직 대화 내용만 텍스트로 출력하세요.
#     """
#     chat = main_chat_model.start_chat(history=st.session_state.chat_history)
#     response = chat.send_message(f"System: {system_prompt}\nUser: {user_message}")
    
#     st.session_state.chat_history.append({"role": "user", "parts": [user_message]})
#     st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
#     return response.text

# def run_combined_tutor(user_message, settings, review_words, db_logs):
#     """[통합 엔진] 변동이 심한 세션 대신 Supabase 실시간 DB 로그를 파싱하여 맥락 100% 반영"""
    
#     # 🌟 언제나 굳건한 DB 실제 로그에서 최근 5개 대화를 추출해 맥락 뼈대를 만듭니다.
#     history_context = ""
#     if db_logs:
#         for log in db_logs[-5:]:
#             role = "AI 선생님" if log['role'] == 'ai' else "학생"
#             history_context += f"{role}: {log['content']}\n"

#     prompt = f"""
#     당신은 {settings['target_language']} 회화 선생님입니다. 
#     단순히 현재 문장만 보지 말고, 제공된 [최근 대화 맥락]의 흐름을 완벽히 파악한 뒤 JSON으로만 응답하세요.

#     [최근 대화 맥락]
#     {history_context}

#     [학생이 방금 한 말]
#     "{user_message}"

#     [미션]
#     1. reply: [최근 대화 맥락]에 이어지는 자연스럽고 다정한 답변 (질문을 포함할 것). 복습 단어 {review_words} 중 하나를 자연스럽게 대화에 녹여서 사용할 것.
#     2. correction: 학생이 방금 한 말이 대화 맥락상 어색하거나 문법이 틀렸다면 '교정 문장 (이유에 대한 한국어 설명)' 형태로 작성. 맥락상 아주 자연스럽고 완벽하면 null 처리.
#     3. new_learning: 이 대화 흐름 속에서 사용하면 좋은 새로운 단어/표현 1개 추천. {{"type":"word", "item":"표현", "meaning":"뜻"}}

#     [응답 양식 - 반드시 JSON만 출력]
#     {{
#         "reply": "선생님의 답변 텍스트",
#         "correction": "교정 내용 또는 null",
#         "new_learning": {{ "type": "word", "item": "item", "meaning": "meaning" }}
#     }}
#     """
    
#     response = main_chat_model.generate_content(prompt)
    
#     try:
#         res_json = json.loads(response.text)
#         # 히스토리 백업용 (기존 규격 유지)
#         st.session_state.chat_history.append({"role": "user", "parts": [user_message]})
#         st.session_state.chat_history.append({"role": "model", "parts": [res_json['reply']]})
#         return res_json
#     except:
#         return {
#             "reply": "I'm sorry, could you say that again?",
#             "correction": None,
#             "new_learning": None
#         }
    
# def get_chat_stream(user_message, settings, review_words, db_logs):
#     """[스트리밍 엔진] 오직 대화에만 집중해서 0.1초 만에 글자를 타타닥 뱉어냅니다."""
#     history_context = ""
#     if db_logs:
#         for log in db_logs[-5:]:
#             role = "AI 선생님" if log['role'] == 'ai' else "학생"
#             history_context += f"{role}: {log['content']}\n"

#     prompt = f"""
#     당신은 {settings['target_language']} 회화 선생님입니다. 
#     아래 [최근 대화 맥락]을 파악하고 학생의 말에 다정하게 대답하세요. 
#     (단, 문법 지적이나 설명은 절대 하지 말고 오직 대화만 이어나갈 것. 질문 포함 권장)

#     [최근 대화 맥락]
#     {history_context}

#     [학생이 방금 한 말]
#     "{user_message}"
#     """
    
#     # 🌟 [에러 해결!] AI 응답 덩어리에서 순수한 '글자(text)'만 쏙쏙 뽑아서 던져줍니다!
#     response = main_chat_model.generate_content(prompt, stream=True)
#     for chunk in response:
#         if chunk.text:
#             yield chunk.text

def get_sync_chat(user_message, settings, db_logs):
    """[동기식 엔진] 스트리밍 없이, 완벽하게 완성된 선생님의 답변 하나를 반환합니다."""
    history_context = ""
    if db_logs:
        for log in db_logs[-5:]:
            role = "AI 선생님" if log['role'] == 'ai' else "학생"
            history_context += f"{role}: {log['content']}\n"

    prompt = f"""
    당신은 {settings['target_language']} 회화 선생님입니다. 
    아래 [최근 대화 맥락]을 파악하고 학생의 말에 다정하게 대답하세요. 
    (단, 문법 지적이나 설명은 절대 하지 말고 오직 대화만 이어나갈 것. 질문 포함 권장. JSON 코드는 절대 출력하지 말고 순수한 텍스트 문장만 출력할 것)

    [최근 대화 맥락]
    {history_context}

    [학생이 방금 한 말]
    "{user_message}"
    """
    res = main_chat_model.generate_content(prompt)
    return res.text.strip()

def run_fast_analyzer(user_message, ai_reply, review_words):
    """[백그라운드 분석기] 대화가 끝난 후, 문법 교정과 단어만 빠르게 JSON으로 뽑아냅니다."""
    prompt = f"""
    학생의 말: "{user_message}"
    선생님의 대답: "{ai_reply}"
    
    위 대화를 바탕으로 아래 JSON만 출력하세요.
    1. correction: 학생의 말이 어색했다면 '교정 문장 (이유)'로 작성. 완벽하면 '완벽합니다!'로 작성.
    2. new_learning: 이 대화 상황에 쓰기 좋은 새로운 표현 1개. {{"type":"word", "item":"표현", "meaning":"뜻"}}
    """
    try:
        res = analyzer_model.generate_content(prompt) # 가벼운 모델 권장
        return json.loads(res.text)
    except:
        return {"correction": None, "new_learning": None}

def generate_quiz_distractors(item, meaning, quiz_type, target_lang):
    """[모델 2] 퀴즈용 그럴싸한 오답 보기 2개 실시간 생성 (Gemini 3.1 Flash Lite)"""
    if quiz_type == "en_to_ko":
        prompt = f"""
        {target_lang} 단어/표현: "{item}" (뜻: {meaning})
        이 단어의 객관식 퀴즈를 만들려고 합니다. 학생들이 적절히 헷갈릴 만한 '그럴싸한 가짜 오답 뜻(한국어)' 2개를 생성하세요.
        단어의 난이도와 품사(동사/명사 등)를 최대한 맞춰야 자연스럽습니다.
        반드시 아래 JSON 형식으로만 응답하세요:
        {{
            "distractors": ["가짜오답1", "가짜오답2"]
        }}
        """
    else: # ko_to_en
        prompt = f"""
        {target_lang} 단어/표현: "{item}" (뜻: {meaning})
        이 단어의 객관식 퀴즈를 만들려고 합니다. 학생들이 적절히 헷갈릴 만한 '그럴싸한 가짜 오답 {target_lang} 단어/표현' 2개를 생성하세요.
        단어의 철자 느낌이나 난이도가 비슷해야 합니다.
        반드시 아래 JSON 형식으로만 응답하세요:
        {{
            "distractors": ["wrong_word1", "wrong_word2"]
        }}
        """
    try:
        response = analyzer_model.generate_content(prompt)
        return json.loads(response.text).get("distractors", [])
    except:
        if quiz_type == "en_to_ko":
            return ["잘못된 보기 A", "잘못된 보기 B"]
        else:
            return ["wrong_option_A", "wrong_option_B"]
        

# def bg_analysis_worker(user_id, log_id, user_msg, review_words):
#     """메인 스트림릿 화면을 절대 얼리지 않고, OS 백그라운드에서 완전히 독립되어 작동하는 진짜 비동기 분석마"""
#     try:
#         analysis = run_background_analysis(user_msg, review_words)
#         corr_text = analysis.get("correction") if analysis.get("correction") else "CLEAN"
        
#         # 1. 문법 교정 결과를 Supabase에 직접 업데이트
#         supabase.table("chat_logs").update({"correction": corr_text}).eq("id", log_id).execute()
        
#         # 2. 신규 단어가 발굴되었다면 단어장에 직접 적재
#         if analysis.get("new_learning") and analysis["new_learning"].get("item"):
#             nl = analysis["new_learning"]
#             supabase.table("vocabulary").insert({
#                 "user_id": user_id, "item_type": nl.get("type", "word"), 
#                 "item": nl["item"], "meaning": nl["meaning"]
#             }).execute()
#     except:
#         # 에러 발생 시 방어벽 처리
#         supabase.table("chat_logs").update({"correction": "CLEAN"}).eq("id", log_id).execute()

# @st.fragment(run_every="2s")
# def render_chat_feed(user_id):
#     """🌟 화면 전체를 흐리게(Blur) 만들지 않고, 2초마다 대화창 내부만 '조용히' 새로고침하는 특수 프래그먼트"""
#     res = supabase.table("chat_logs").select("*").eq("user_id", user_id).order("id").execute()
#     for log in res.data[-10:]: 
#         with st.chat_message(log['role']):
#             st.write(log['content'])
#             if log.get('correction') and log['correction'] not in ['PENDING', 'CLEAN']:
#                 st.markdown(f'<div class="correction-box">💡 <b>더 자연스러운 표현:</b><br>{log["correction"]}</div>', unsafe_allow_html=True)
#             elif log.get('correction') == 'PENDING':
#                 st.caption("⚡ 더 자연스러운 표현 분석 중...")


# ==========================================
# 3. 화면 UI 및 탭 기능 분기
# ==========================================

# --- 1단계: 로그인 / 회원가입 화면 (이메일 & 초대 코드 보안 적용) ---
if st.session_state.user_id is None:
    st.title("🔐 AI Tutor 로그인")
    st.write("초대 코드가 있어야만 접속할 수 있는 프라이빗 학습 공간입니다.")
    
    with st.form("login_form"):
        # 🌟 이름 대신 이메일 입력으로 변경
        user_email = st.text_input("이메일", placeholder="예: user@example.com")
        
        # 비밀번호(초대코드) 입력칸
        invite_code = st.text_input("초대 코드 (비밀번호)", type="password", placeholder="초대 코드를 입력하세요")
        
        submit_btn = st.form_submit_button("입장하기 🚀", use_container_width=True)
        
        if submit_btn:
            if not user_email.strip():
                st.warning("⚠️ 이메일을 입력해 주세요!")
            # 금고(secrets)에 있는 코드와 입력한 코드가 다르면 차단!
            elif invite_code != st.secrets["INVITE_CODE"]:
                st.error("🚨 초대 코드가 일치하지 않습니다. 관리자에게 문의하세요.")
            else:
                with st.spinner("로그인 중..."):
                    # 🌟 DB에서 username이 아닌 'email' 컬럼을 검색합니다!
                    existing_user = supabase.table("users").select("*").eq("email", user_email).execute()
                    
                    if existing_user.data:
                        # 기존 유저면 로그인 성공
                        st.session_state.user_id = existing_user.data[0]['id']
                    else:
                        # 신규 유저면 새 계정 생성 (email 저장)
                        new_user = supabase.table("users").insert({"email": user_email}).execute()
                        st.session_state.user_id = new_user.data[0]['id']
                        
                    st.success("환영합니다! 🎉")
                    st.rerun()

# 🌟 유저가 로그인을 통과했다면 (기존 코드 유지)
else:
    logs_res = supabase.table("chat_logs").select("*").eq("user_id", st.session_state.user_id).order("id").execute()
    global_vocab_res = supabase.table("vocabulary").select("*").eq("user_id", st.session_state.user_id).order("id", desc=True).execute()
    
    # --- 2단계: 신규 유저 온보딩 화면 (대화 기록이 0개일 때) ---
    if not logs_res.data:
        st.title("🎉 환영합니다! 첫 대화 전에 설정을 맞춰볼까요?")
        st.info("선생님의 성향과 학습 목표를 알려주시면, 선생님이 먼저 반갑게 인사를 건넬 거예요!")
        
        with st.form("onboarding_form"):
            new_lang = st.selectbox("어떤 언어를 배우고 싶나요?", ["영어", "일본어", "중국어", "한국어"])
            new_sit = st.selectbox("어떤 상황을 연습할까요?", ["일상 대화", "비즈니스 미팅", "해외 여행", "공항 입국심사", "카페 주문"])
            new_gen = st.radio("원하는 선생님 성별은?", ["여성", "남성"])
            new_level = st.slider("나의 현재 실력은? (1:초보 ~ 5:원어민)", 1, 5, value=1)
            
            if st.form_submit_button("설정 완료하고 대화 시작하기!"):
                with st.spinner("AI 선생님이 맞춤형 첫 인사를 준비하고 있습니다..."):
                    supabase.table("users").update({
                        "target_language": new_lang, "situation": new_sit, "partner_gender": new_gen, "current_level": new_level
                    }).eq("id", st.session_state.user_id).execute()
                    
                    first_prompt = f"당신은 {new_lang} 회화 선생님({new_gen})입니다. 상황은 '{new_sit}'이며 학생 레벨은 {new_level}입니다. 반갑게 인사를 건네고 대답하기 쉬운 가벼운 첫 질문을 던지세요. (대사만 출력)"
                    ai_first_msg = main_chat_model.generate_content(first_prompt).text
                    
                    supabase.table("chat_logs").insert({"user_id": st.session_state.user_id, "role": "ai", "content": ai_first_msg}).execute()
                    st.rerun()

    # --- 3단계: 기존 유저 메인 화면 (들여쓰기 및 동기화 최적화 완료) ---
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["💬 대화", "📚 단어장 & 발음", "📝 기록", "⚙️ 설정"])

        # 유저 세팅 가져오기
        user_settings = supabase.table("users").select("*").eq("id", st.session_state.user_id).execute().data[0]
        settings = {
            "target_language": user_settings.get("target_language", "영어"),
            "situation": user_settings.get("situation", "일상 대화"),
            "partner_gender": user_settings.get("partner_gender", "여성"),
            "level": user_settings.get("current_level", 1)
        }
        
        # 🌟 [신규] 지연 로딩용 복습 단어 및 백그라운드 분석 엔진
        review_words = [v['item'] for v in global_vocab_res.data[:3]]
        pending_logs = [log for log in logs_res.data if log['role'] == 'user' and log.get('correction') == 'PENDING']
        
        if pending_logs:
            target_log = pending_logs[-1] # 가장 최근 대기 항목 저격
            # 사용자가 답변을 읽는 동안 뒤에서 무거운 분석기 실행!
            analysis = run_background_analysis(target_log['content'], review_words)
            corr_text = analysis.get("correction") if analysis.get("correction") else "CLEAN"
            
            # 분석 완료 후 DB 업데이트 (PENDING -> 교정내용 혹은 CLEAN으로 변경)
            supabase.table("chat_logs").update({"correction": corr_text}).eq("id", target_log['id']).execute()
            
            # 신규 단어 저장 동기화
            if analysis.get("new_learning") and analysis["new_learning"].get("item"):
                nl = analysis["new_learning"]
                supabase.table("vocabulary").insert({"user_id": st.session_state.user_id, "item_type": nl.get("type", "word"), "item": nl["item"], "meaning": nl["meaning"]}).execute()
                st.session_state.quiz_word = None
            st.rerun()


        # -----------------------------------------------------
        # TAB 1: 실시간 대화 (마이크 잠금 UX & 개별 문장 듣기 버튼 적용)
        # -----------------------------------------------------
        with tab1:
            st.subheader(f"🗣️ {settings['situation']} (Level {settings['level']})")
            
            review_words = [v['item'] for v in global_vocab_res.data[:3]]
            chat_mode = st.radio("대화 방식을 선택하세요:", ["💬 텍스트 채팅 모드", "🎙️ 실시간 말하기 모드"], horizontal=True, key="chat_interface_selector")
            st.write("---")

            # 🌟 1. 대화 피드 출력 (AI 답변에 개별 듣기 버튼 추가)
            for log in logs_res.data[-10:]: 
                with st.chat_message(log['role']):
                    st.write(log['content'])
                    
                    # AI 선생님의 답변일 경우 문장 아래에 [발음 듣기] 버튼 생성
                    if log['role'] == 'ai' and chat_mode == "🎙️ 실시간 말하기 모드":
                        # 버튼 키(key)를 고유한 log ID로 설정하여 각각 독립적으로 작동하게 만듦
                        if st.button("🔊 발음 듣기", key=f"play_btn_{log['id']}"):
                            lang_map = {"영어": "en", "일본어": "ja", "중국어": "zh-CN", "한국어": "ko"}
                            tts_lang = lang_map.get(settings['target_language'], "en")
                            tts = gTTS(text=log['content'], lang=tts_lang)
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            # 버튼을 누른 순간에만 플레이어가 나타나며 즉시 재생됨
                            st.audio(fp.getvalue(), format="audio/mp3", autoplay=True)
                    
                    # 유저 문장일 경우 교정 내역 표시
                    if log.get('correction') and log['correction'] not in ['PENDING', 'CLEAN']:
                        st.markdown(f'<div class="correction-box">💡 <b>더 자연스러운 표현:</b><br>{log["correction"]}</div>', unsafe_allow_html=True)

            st.write("---")
            user_input = None
            
            # --- 분기 A: 💬 텍스트 채팅 모드 ---
            if chat_mode == "💬 텍스트 채팅 모드":
                if msg := st.chat_input("선생님에게 메시지를 입력하세요 (Enter)"):
                    user_input = msg
                    with st.chat_message("user"):
                        st.write(user_input)
            
            # --- 분기 B: 🎙️ 실시간 말하기 모드 ---
            else:
                # 🌟 2. 마이크 숨김 & 락 해제 UX
                if st.session_state.get("autoplay_tts"):
                    st.success("🔊 선생님의 답변이 도착했습니다! 끝까지 듣고 아래 버튼을 눌러주세요.")
                    st.audio(st.session_state.autoplay_tts, format="audio/mp3", autoplay=True)
                    
                    if st.button("🎤 다 들었어요! 이제 제가 말할게요", use_container_width=True, type="primary"):
                        # 이제 latest_audio로 넘길 필요 없이 그냥 비워버리면 끝납니다!
                        st.session_state.autoplay_tts = None
                        st.rerun()
                
                # 🌟 3. 마이크가 켜지는 구역 (락이 풀렸을 때)
                else:
                    st.markdown('<div class="speaking-guide"><b>🎙️ 이제 마이크를 켜고 편하게 대답해 보세요.</b></div>', unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        audio_bytes = audio_recorder(text="클릭하여 말하기 시작", recording_color="#ef4444", neutral_color="#3b82f6", icon_size="2x", key="speaking_mode_voice")
                    
                    if audio_bytes and st.session_state.get("last_speaking_audio") != audio_bytes:
                        st.session_state.last_speaking_audio = audio_bytes
                        with st.spinner("🎧 목소리를 분석하여 문장으로 변환하는 중..."):
                            stt_prompt = f"""
                            당신은 음성 인식(STT) 기계입니다. 사용자가 말한 {settings['target_language']} 음성을 텍스트로만 변환하세요.
                            [절대 규칙]
                            1. 어떤 경우에도 JSON 형식({{"text": "..."}})을 출력하지 마세요. 오직 순수 텍스트만 출력하세요.
                            2. 헛기침, 잡음, 짧은 숨소리 등으로 판단되어 변환할 말이 없다면 무조건 대문자로 "SILENCE" 라고만 출력하세요.
                            3. "Oh", "Uh" 같은 무의미한 감탄사 하나만 들리면 "SILENCE"로 처리하세요.
                            """
                            try:
                                stt_res = audio_coach_model.generate_content([stt_prompt, {"mime_type": "audio/wav", "data": audio_bytes}])
                                parsed_text = stt_res.text.strip()
                                
                                if "SILENCE" in parsed_text.upper() or "{" in parsed_text or "}" in parsed_text or parsed_text.lower() in ["oh", "oh.", '"oh"', "'oh'"]:
                                    st.warning("⚠️ 목소리가 명확히 인식되지 않았습니다. 다시 또렷하게 말씀해 주세요!")
                                    st.session_state.voice_stt_preview = None
                                else:
                                    st.session_state.voice_stt_preview = parsed_text
                                    
                            except Exception as e:
                                if "429" in str(e) or "ResourceExhausted" in str(e):
                                    st.error("⏳ 구글 AI 무료 사용량(1분 한도)을 초과했습니다. 잠시 후 다시 시도해주세요.")
                                else:
                                    st.error("⚠️ 일시적인 오류가 발생했습니다. 다시 시도해 주세요.")
                                st.session_state.voice_stt_preview = None

                    if st.session_state.voice_stt_preview:
                        st.markdown(f"""
                        <div style="border: 2px dashed #3B82F6; background-color: #F8FAFC; padding: 15px; border-radius: 12px; margin-top:15px;">
                            <span style="font-size:12px; color:#475569; font-weight:bold;">🔍 전송 전 받아쓰기 최종 확인:</span>
                            <h4 style="color:#1E3A8A; margin: 5px 0 0 0;">"{st.session_state.voice_stt_preview}"</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(" ")
                        
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("🚀 이대로 선생님께 전송", use_container_width=True, type="primary"):
                                user_input = st.session_state.voice_stt_preview
                                st.session_state.voice_stt_preview = None
                        with btn_col2:
                            if st.button("🔄 지우고 다시 녹음하기", use_container_width=True):
                                st.session_state.voice_stt_preview = None
                                st.rerun()

            # 🚀 4. 대화 전송 및 데이터 저장 구역
            if user_input:
                with st.spinner("선생님이 답변을 작성하고 원어민 발음을 준비하는 중입니다... ⏳"):
                    ai_reply_text = get_sync_chat(user_input, settings, logs_res.data)
                    analysis_result = run_fast_analyzer(user_input, ai_reply_text, review_words)
                    
                    supabase.table("chat_logs").insert({"user_id": st.session_state.user_id, "role": "user", "content": user_input, "correction": analysis_result['correction']}).execute()
                    supabase.table("chat_logs").insert({"user_id": st.session_state.user_id, "role": "ai", "content": ai_reply_text}).execute()
                    
                    if analysis_result.get("new_learning") and analysis_result["new_learning"].get("item"):
                        nl = analysis_result["new_learning"]
                        supabase.table("vocabulary").insert({"user_id": st.session_state.user_id, "item_type": nl.get("type", "word"), "item": nl["item"], "meaning": nl["meaning"]}).execute()
                        st.session_state.quiz_word = None
                    
                    # 답변이 생성되면 자동 재생을 위해 음성을 장전합니다!
                    if chat_mode == "🎙️ 실시간 말하기 모드":
                        lang_map = {"영어": "en", "일본어": "ja", "중국어": "zh-CN", "한국어": "ko"}
                        tts = gTTS(text=ai_reply_text, lang=lang_map.get(settings['target_language'], "en"))
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        st.session_state.autoplay_tts = fp.getvalue()
                
                st.rerun()

        # # -----------------------------------------------------
        # # TAB 1: 실시간 대화 (초고속 대화 & 백그라운드 자동 피드백)
        # # -----------------------------------------------------
        # with tab1:
        #     st.subheader(f"🗣️ {settings['situation']} (Level {settings['level']})")
            
        #     # 복습용 목표 상위 3개 단어 바인딩 (실시간 동기화 리스트 활용)
        #     review_words = [v['item'] for v in global_vocab_res.data[:3]]

        #     # 🌟 [기존 유지] 대화 모드 체인저 선언
        #     chat_mode = st.radio("대화 방식을 선택하세요:", ["💬 텍스트 채팅 모드", "🎙️ 실시간 말하기 모드"], horizontal=True, key="chat_interface_selector")
        #     st.write("---")

        #     # 🌟 [기존 유지] 브라우저 자동 재생 차단벽을 허무는 Base64 오디오 인젝터
        #     if chat_mode == "🎙️ 실시간 말하기 모드" and st.session_state.autoplay_tts:
        #         import base64
        #         b64_audio = base64.b64encode(st.session_state.autoplay_tts).decode()
        #         st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3"></audio>', unsafe_allow_html=True)
        #         st.session_state.autoplay_tts = None  # 단발성 재생 후 즉시 소거

        #     # 🌟 [개선] 비동기 에러를 유발하던 fragment 대신, 선명하고 버그 없는 순정 피드 출력
        #     for log in logs_res.data[-10:]: 
        #         with st.chat_message(log['role']):
        #             st.write(log['content'])
        #             # 원샷 엔진이므로 이제 'PENDING' 없이 정답과 교정이 동시에 깨끗하게 출력됩니다.
        #             if log.get('correction') and log['correction'] not in ['PENDING', 'CLEAN']:
        #                 st.markdown(f'<div class="correction-box">💡 <b>더 자연스러운 표현:</b><br>{log["correction"]}</div>', unsafe_allow_html=True)

        #     st.write("---")
            
        #     # --- 분기 A: 💬 텍스트 채팅 모드 ---
        #     if chat_mode == "💬 텍스트 채팅 모드":
        #         if text_msg := st.chat_input("선생님에게 메시지를 입력하세요 (Enter)"):
        #             with st.chat_message("user"):
        #                 st.write(text_msg)
                    
        #             with st.spinner("선생님이 답변을 생각하는 중입니다..."):
        #                 result = run_combined_tutor(text_msg, settings, review_words, logs_res.data)

        #             # 결과에서 나온 문법 교정(correction) 데이터를 유저 로그와 즉시 결합하여 저장
        #             supabase.table("chat_logs").insert({
        #                 "user_id": st.session_state.user_id, "role": "user", "content": text_msg, "correction": result['correction']
        #             }).execute()
                    
        #             # AI 답변 즉시 저장
        #             supabase.table("chat_logs").insert({
        #                 "user_id": st.session_state.user_id, "role": "ai", "content": result['reply']
        #             }).execute()
                    
        #             # 실시간 추천 표현 단어장 즉시 동기화 적재
        #             if result.get("new_learning") and result["new_learning"].get("item"):
        #                 nl = result["new_learning"]
        #                 supabase.table("vocabulary").insert({"user_id": st.session_state.user_id, "item_type": nl.get("type", "word"), "item": nl["item"], "meaning": nl["meaning"]}).execute()
        #                 st.session_state.quiz_word = None  # 퀴즈 리스트 리프레시 트리거
                    
        #             st.rerun()

        #     # --- 분기 B: 🎙️ 실시간 말하기 모드 (기존 요구사항 및 디테일 100% 반영) ---
        #     else:
        #         st.markdown("""
        #         <div class="speaking-guide">
        #             <b>🎙️ 실시간 음성 회화 안심 가이드</b><br>
        #             1. 아래 마이크를 누르면 <b>[붉은색]</b>으로 변경되며 음성 녹음이 개시됩니다.<br>
        #             2. 문장 도중 중간중간 더듬거나 말을 멈추셔도 괜찮으니 마음 편히 완창하세요.<br>
        #             3. 말씀이 모두 끝나셨다면 마이크를 <b>[한 번 더 눌러서]</b> 확실하게 꺼주세요!
        #         </div>
        #         """, unsafe_allow_html=True)

        #         col1, col2, col3 = st.columns([1, 2, 1])
        #         with col2:
        #             audio_bytes = audio_recorder(text="클릭하여 말하기 시작", recording_color="#ef4444", neutral_color="#3b82f6", icon_size="2x", key="speaking_mode_voice")
                
        #         if audio_bytes and st.session_state.get("last_speaking_audio") != audio_bytes:
        #             st.session_state.last_speaking_audio = audio_bytes
        #             with st.spinner("🎧 목소리를 분석하여 문장으로 변환하는 중..."):
                        
        #                 # 🌟 [디테일 유지] 유저님이 작성하신 강력한 환각 및 소음 방지 프롬프트 세팅
        #                 stt_prompt = f"""
        #                 이 오디오 파일에서 사용자가 말한 {settings['target_language']} 문장을 그대로 텍스트로만 받아적어주세요. 
        #                 [필수 주의사항]
        #                 1. 만약 오디오가 비어있거나, 사람의 목소리가 명확하게 들리지 않거나, 단순한 소음/숨소리만 있다면 **절대로 문장을 지어내지 말고 딱 한 단어 "SILENCE" 라고만 출력**하세요.
        #                 2. 사소하게 말을 더듬거나 어휘를 반복한 구간이 있다면 문맥에 맞게 매끄러운 1개의 문장으로 정제하세요.
        #                 """
                        
        #                 stt_res = audio_coach_model.generate_content([stt_prompt, {"mime_type": "audio/wav", "data": audio_bytes}])
        #                 parsed_text = stt_res.text.strip()
                        
        #                 if "SILENCE" in parsed_text.upper():
        #                     st.warning("⚠️ 목소리가 명확히 인식되지 않았습니다. 마이크를 켜고 다시 또렷하게 말씀해 주세요!")
        #                     st.session_state.voice_stt_preview = None
        #                 else:
        #                     st.session_state.voice_stt_preview = parsed_text
                
        #         # 🌟 [디테일 유지] 전송 전 받아쓰기 샌드박스 검토 UI 구역
        #         if st.session_state.voice_stt_preview:
        #             st.markdown(f"""
        #             <div style="border: 2px dashed #3B82F6; background-color: #F8FAFC; padding: 15px; border-radius: 12px; margin-top:15px;">
        #                 <span style="font-size:12px; color:#475569; font-weight:bold;">🔍 전송 전 받아쓰기 최종 확인:</span>
        #                 <h4 style="color:#1E3A8A; margin: 5px 0 0 0;">"{st.session_state.voice_stt_preview}"</h4>
        #             </div>
        #             """, unsafe_allow_html=True)
                    
        #             st.write(" ")
        #             btn_col1, btn_col2 = st.columns(2)
                    
        #             # 버튼 1: 이대로 전송하기 (원샷 결합 완료)
        #             with btn_col1:
        #                 if st.button("🚀 이대로 선생님께 전송", use_container_width=True, type="primary"):
        #                     final_msg = st.session_state.voice_stt_preview
                            
        #                     with st.spinner("선생님이 대답을 생각하고 원어민 발음을 생성하는 중..."):
        #                         # 🌟 여기도 맨 끝에 logs_res.data 를 쥐여줍니다.
        #                         result = run_combined_tutor(final_msg, settings, review_words, logs_res.data)
                                
        #                         # 🌟 [디테일 유지] 유저님이 설계하신 4개 국어 음성 언어팩 매핑 시스템
        #                         lang_map = {"영어": "en", "일본어": "ja", "중국어": "zh-CN", "한국어": "ko"}
        #                         tts_lang = lang_map.get(settings['target_language'], "en")
                                
        #                         tts = gTTS(text=result['reply'], lang=tts_lang)
        #                         fp = io.BytesIO()
        #                         tts.write_to_fp(fp)
        #                         st.session_state.autoplay_tts = fp.getvalue()
                                
        #                     # 수집된 통합 분석 데이터를 결합하여 유저 로그 적재
        #                     supabase.table("chat_logs").insert({"user_id": st.session_state.user_id, "role": "user", "content": final_msg, "correction": result['correction']}).execute()
        #                     # AI 답변 적재
        #                     supabase.table("chat_logs").insert({"user_id": st.session_state.user_id, "role": "ai", "content": result['reply']}).execute()
                            
        #                     # 새 단축 표현 동기화 적재
        #                     if result.get("new_learning") and result["new_learning"].get("item"):
        #                         nl = result["new_learning"]
        #                         supabase.table("vocabulary").insert({"user_id": st.session_state.user_id, "item_type": nl.get("type", "word"), "item": nl["item"], "meaning": nl["meaning"]}).execute()
        #                         st.session_state.quiz_word = None
                                
        #                     st.session_state.voice_stt_preview = None
        #                     st.rerun()
                    
        #             # 버튼 2: 지우고 다시 녹음하기
        #             with btn_col2:
        #                 if st.button("🔄 지우고 다시 녹음하기", use_container_width=True):
        #                     st.session_state.voice_stt_preview = None
        #                     st.rerun()

        #     # -----------------------------------------------------
        #     # [3단계] 🌟 [누락되었던 핵심] 백그라운드 지연 로딩 감시자 엔진
        #     # -----------------------------------------------------
        #     # 화면에 대화 내용을 먼저 다 띄워준 직후, 대기 중인 PENDING 항목이 있는지 체크합니다.
        #     pending_logs = [log for log in logs_res.data if log['role'] == 'user' and log.get('correction') == 'PENDING']
            
        #     if pending_logs:
        #         target_log = pending_logs[-1]  # 가장 최신 전송 문장
                
        #         # 사용자가 AI 선생님의 답변을 읽거나 음성을 듣는 동안 화면 밑에서 조용히 분석 실행
        #         with st.spinner("선생님이 더 자연스러운 표현을 고민하고 있습니다... ⚡"):
        #             analysis = run_background_analysis(target_log['content'], review_words)
                    
        #             corr_text = analysis.get("correction") if analysis.get("correction") else "CLEAN"
        #             # 분석 완료된 데이터를 DB에 업데이트하여 PENDING 간판을 내림
        #             supabase.table("chat_logs").update({"correction": corr_text}).eq("id", target_log['id']).execute()
                    
        #             # 새 표현이 발굴되었다면 단어장에 자동 동기화 적재
        #             if analysis.get("new_learning") and analysis["new_learning"].get("item"):
        #                 nl = analysis["new_learning"]
        #                 supabase.table("vocabulary").insert({
        #                     "user_id": st.session_state.user_id, "item_type": nl.get("type", "word"), "item": nl["item"], "meaning": nl["meaning"]
        #                 }).execute()
        #                 st.session_state.quiz_word = None # 단어장 리스트 동기화 유도
                
        #         # 🌟 데이터 처리가 완수되었으므로 자동으로 앱을 새로고침하여 노란 교정 상자를 출력!
        #         st.rerun()


        # -----------------------------------------------------
        # TAB 2: 단어장 & 발음 훈련 & 퀴즈 통합 기능 (글로벌 실시간 반영)
        # -----------------------------------------------------
        with tab2:
            st.subheader("📚 내 스마트 단어장")
            
            # --- 🎯 퀴즈 카드 섹션 (중복 금지 + 화살표 이동 + 하이브리드 고속 엔진) ---
            with st.expander("🧩 단어 퀴즈 풀기 (실력 점검!)", expanded=True):
                # 최신 전역 스냅샷 기반 필터링
                learning_vocab = [v for v in global_vocab_res.data if v.get('success_count', 0) < 3]
                
                if not learning_vocab:
                    st.info("퀴즈를 출제할 단어가 없습니다. 대화를 통해 단어를 먼저 수집해 보세요!")
                else:
                    if st.session_state.quiz_word is None:
                        import random
                        # 단어 중복 방지
                        candidates = [v for v in learning_vocab if v['id'] not in st.session_state.quiz_history]
                        if not candidates:
                            st.session_state.quiz_history = []
                            candidates = learning_vocab
                        if len(learning_vocab) > 1 and len(st.session_state.quiz_history) > 0:
                            candidates = [v for v in candidates if v['id'] != st.session_state.quiz_history[-1]]
                        
                        selected_word = random.choice(candidates)
                        st.session_state.quiz_word = selected_word
                        st.session_state.quiz_history.append(selected_word['id'])
                        
                        # 유형 순화 덱 빌드
                        if not st.session_state.quiz_type_pool:
                            st.session_state.quiz_type_pool = ["en_to_ko", "ko_to_en", "typing"]
                            random.shuffle(st.session_state.quiz_type_pool)
                        st.session_state.quiz_type = st.session_state.quiz_type_pool.pop()
                        
                        # 오답 선지 로딩 분기 (하이브리드 패스트)
                        if st.session_state.quiz_type != "typing":
                            if len(global_vocab_res.data) >= 3:
                                if st.session_state.quiz_type == "en_to_ko":
                                    wrong_options = [v['meaning'] for v in global_vocab_res.data if v['id'] != selected_word['id']]
                                    wrong_samples = random.sample(wrong_options, 2)
                                    options = wrong_samples + [selected_word['meaning']]
                                else:
                                    wrong_options = [v['item'] for v in global_vocab_res.data if v['id'] != selected_word['id']]
                                    wrong_samples = random.sample(wrong_options, 2)
                                    options = wrong_samples + [selected_word['item']]
                            else:
                                with st.spinner("초기 퀴즈 세팅 중..."):
                                    wrong_samples = generate_quiz_distractors(
                                        selected_word['item'], selected_word['meaning'], st.session_state.quiz_type, settings['target_language']
                                    )
                                if st.session_state.quiz_type == "en_to_ko":
                                    options = wrong_samples + [selected_word['meaning']]
                                else:
                                    options = wrong_samples + [selected_word['item']]
                            
                            random.shuffle(options)
                            st.session_state.quiz_options = options
                        
                        st.session_state.quiz_feedback = None
                        st.session_state.quiz_answered = False
                        st.rerun()
                    
                    if st.session_state.quiz_type == "en_to_ko":
                        card_title = "💚 영어를 보고 뜻을 고르세요"
                        question_word = st.session_state.quiz_word['item']
                        correct_answer = st.session_state.quiz_word['meaning']
                    elif st.session_state.quiz_type == "ko_to_en":
                        card_title = "⚡ 뜻을 보고 알맞은 영어를 고르세요"
                        question_word = st.session_state.quiz_word['meaning']
                        correct_answer = st.session_state.quiz_word['item']
                    else:
                        card_title = "⌨️ 빈칸 채우기! 영어로 직접 타이핑하세요"
                        question_word = st.session_state.quiz_word['meaning']
                        correct_answer = st.session_state.quiz_word['item']

                    st.markdown(f"""
                    <div style="background-color: #F3F4F6; padding: 22px; border-radius: 18px; border: 2px solid #10B981; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                        <span style="font-size: 13px; color: #059669; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">{card_title}</span>
                        <h2 style="color: #111827; margin: 12px 0; font-size: 26px; font-weight: 800;">{question_word}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(" ")
                    
                    # [1 상태] 문제 풀기 전
                    if not st.session_state.quiz_answered:
                        with st.form("quiz_form"):
                            if st.session_state.quiz_type == "typing":
                                # 🌟 [개선] 글자별로 검사하여 특수문자('.', "'", ',')는 그대로 노출하는 힌트 생성기
                                hint_words = []
                                for w in correct_answer.split():
                                    if len(w) > 1:
                                        word_hint = w[0] # 첫 글자는 무조건 오픈
                                        for char in w[1:]:
                                            if char.isalnum(): # 알파벳이나 숫자라면 빈칸 처리
                                                word_hint += " _"
                                            else: # 특수문자나 문장기호라면 그대로 표시!
                                                word_hint += f" {char}"
                                        hint_words.append(word_hint)
                                    else:
                                        hint_words.append(w)
                                
                                display_hint = "   ".join(hint_words)
                                
                                st.markdown(f"""
                                <div style="background-color: #EFF6FF; padding: 12px 15px; border-radius: 10px; border: 1px solid #BFDBFE; margin-bottom: 15px;">
                                    <span style="color: #1E40AF; font-size: 12px; font-weight: bold;">🧩 내가 방금 배운 단어 힌트:</span><br>
                                    <code style="font-size: 18px; color: #1D4ED8; font-weight: 800; letter-spacing: 2px;">{display_hint}</code>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                user_choice = st.text_input("위 힌트에 맞게 정답 표현을 입력하세요:", placeholder="여기에 타이핑하세요...")
                            
                            # ... (이하 st.radio 및 제출 버튼 로직은 기존과 동일) ...
                            else:
                                user_choice = st.radio("문맥에 맞는 정답은?", st.session_state.quiz_options, key=f"quiz_radio_{st.session_state.quiz_word['id']}")
                            
                            if st.form_submit_button("정답 확인하기 ✔️"):
                                st.session_state.quiz_answered = True
                                
                                if st.session_state.quiz_type == "typing":
                                    # 양끝 공백을 지우고 대소문자 구분을 없애서 채점 정확도 상향
                                    is_correct = user_choice.strip().lower() == correct_answer.strip().lower()
                                else:
                                    is_correct = (user_choice == correct_answer)

                                if is_correct:
                                    st.session_state.quiz_feedback = "정답"
                                    c_success = st.session_state.quiz_word.get('success_count', 0) + 1
                                    supabase.table("vocabulary").update({"success_count": c_success}).eq("id", st.session_state.quiz_word['id']).execute()
                                else:
                                    st.session_state.quiz_feedback = "오답"
                                    c_fail = st.session_state.quiz_word.get('fail_count', 0) + 1
                                    supabase.table("vocabulary").update({"fail_count": c_fail}).eq("id", st.session_state.quiz_word['id']).execute()
                                st.rerun()
                                
                    # 🌟 else의 위치와 내부 실행 코드들의 시작 라인을 일렬로 정렬했습니다.
                    else:
                        if st.session_state.quiz_feedback == "정답":
                            st.success(f"🎉 완벽해요! 정답입니다.\n\n**{st.session_state.quiz_word['item']}** : {st.session_state.quiz_word['meaning']}")
                        else:
                            st.error(f"❌ 틀렸습니다! 정답은 **[ {correct_answer} ]** 입니다.")
                        
                        st.write(" ")
                        if st.button("다음 문제 넘어가기 ➡️", use_container_width=True):
                            st.session_state.quiz_word = None
                            st.session_state.quiz_answered = False
                            st.rerun()

            st.write("---")

            # 전역 변수에서 바인딩 분리 (최신 데이터 순으로 노출)
            active_words = [v for v in global_vocab_res.data if v.get('success_count', 0) < 3]
            mastered_words = [v for v in global_vocab_res.data if v.get('success_count', 0) >= 3]

            st.markdown("### 🔥 현재 학습 중인 단어")
            if not active_words:
                st.caption("현재 학습 중인 단어가 없습니다.")
            
            for v in active_words:
                danger_badge = "⚠️ 집중 요망! (자주 틀림)" if v.get('fail_count', 0) >= 2 else ""
                badge = f"🔥 집중 복습 | {danger_badge}" if v['is_pinned'] else f"{v['item_type'].upper()} {danger_badge}"
                box_class = "pinned-word" if v['is_pinned'] else "normal-word"
                title_color = "#B91C1C" if v['is_pinned'] else "#1F2937"
                
                col_text, col_audio, col_del = st.columns([3, 0.6, 0.6])
                with col_text:
                    st.markdown(f"""
                    <div class="{box_class}" style="margin-bottom:0;">
                        <h4 style="color:{title_color}; margin:0;">{v['item']} <span style="font-size:11px; font-weight:normal; color:#ef4444;">[{badge}]</span></h4>
                        <p style="margin:5px 0 0 0; color:#4B5563; font-size:14px;">{v['meaning']} (틀린횟수: {v.get('fail_count', 0)}회)</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_audio:
                    if st.button("🔊", key=f"tts_{v['id']}"):
                        tts = gTTS(text=v['item'], lang='en' if settings['target_language']=='영어' else 'ja')
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        st.audio(fp.getvalue(), format="audio/mp3", autoplay=True)
                with col_del:
                    if st.button("🗑️", key=f"del_{v['id']}"):
                        supabase.table("vocabulary").delete().eq("id", v['id']).execute()
                        st.toast("단어가 삭제되었습니다.")
                        st.rerun()

                with st.expander(f"🎤 이 표현 발음 연습하기"):
                    rec_col, eval_col = st.columns([1, 2])
                    with rec_col:
                        audio_bytes = audio_recorder(text="눌러서 녹음", recording_color="#e84118", neutral_color="#00a8ff", icon_size="2x", key=f"rec_{v['id']}")
                    with eval_col:
                        if audio_bytes and st.button("평가받기", key=f"eval_btn_{v['id']}"):
                            with st.spinner("분석 중..."):
                                res = run_voice_coach(v['item'], audio_bytes)
                                st.session_state.pronunciation_evals[v['id']] = res
                    
                    if v['id'] in st.session_state.pronunciation_evals:
                        res = st.session_state.pronunciation_evals[v['id']]
                        st.metric(label="🎖️ 점수", value=f"{res['score']}점")
                        st.success(f"👼 피드백: {res['feedback']}")
                        st.info(f"💡 꿀팁: {res['correction_tip']}")
                st.write("---")

            st.write(" ")
            st.markdown("### 👑 명예의 전당 (완벽히 마스터한 표현)")
            
            if not mastered_words:
                st.caption("아직 마스터한 단어가 없습니다.")
            for v in mastered_words:
                st.markdown(f"""
                <div style="border: 2px solid #10B981; background-color: #ECFDF5; padding: 12px; border-radius: 10px; margin-bottom: 5px;">
                    <h5 style="color:#065F46; margin:0;">✨ {v['item']}</h5>
                    <p style="margin:5px 0 0 0; color:#047857; font-size:13px;">{v['meaning']} (완벽 숙지 완료!)</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_audio, col_del = st.columns([1, 1])
                with col_audio:
                    if st.button("🔊 발음 듣기", key=f"tts_master_{v['id']}"):
                        tts = gTTS(text=v['item'], lang='en' if settings['target_language']=='영어' else 'ja')
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        st.audio(fp.getvalue(), format="audio/mp3", autoplay=True)
                with col_del:
                    if st.button("🗑️ 보관함에서 삭제", key=f"del_master_{v['id']}"):
                        supabase.table("vocabulary").delete().eq("id", v['id']).execute()
                        st.toast("보관함에서 삭제되었습니다.")
                        st.rerun()


        # -----------------------------------------------------
        # TAB 3: 대화 기록 (한국 시간 보정 + 아코디언 정렬형)
        # -----------------------------------------------------
        with tab3:
            st.subheader("📝 최근 대화 기록")
            
            # -----------------------------------------------------
            # TAB 3: 대화 기록 렌더링 구역
            # -----------------------------------------------------
            grouped_logs = defaultdict(list)
            for h in logs_res.data:
                # 🌟 [수정] 어떤 포맷이나 None이 와도 절대 터지지 않는 무적 시간 파싱
                raw_time = h.get('created_at')
                
                if not raw_time:
                    # 1. 시간이 아직 안 적혀서 None인 경우, 현재 한국 시간으로 임시 대체
                    kst_time = datetime.datetime.now()
                else:
                    try:
                        # 2. 표준 ISO 포맷 파싱 시도 (공백 및 Z 치환 처리)
                        clean_time = raw_time.replace('Z', '+00:00').replace(' ', 'T')
                        if '+' in clean_time and clean_time.endswith('+00'):
                            clean_time += ':00' # 일부 타임존 생략 대응
                        utc_time = datetime.datetime.fromisoformat(clean_time)
                        kst_time = utc_time + timedelta(hours=9)
                    except Exception:
                        try:
                            # 3. 만약 포맷이 이상해서 실패하면 앞글자(날짜/시간)만 강제로 잘라서 파싱
                            clean_time = raw_time.replace(' ', 'T')[:19]
                            kst_time = datetime.datetime.strptime(clean_time, "%Y-%m-%dT%H:%M:%S") + timedelta(hours=9)
                        except Exception:
                            # 4. 최후의 보루: 이마저도 안 되면 그냥 현재 시간으로 방어
                            kst_time = datetime.datetime.now()
                
                # 정렬 및 뷰 바인딩 진행
                date_str = kst_time.strftime("%Y년 %m월 %d일")
                grouped_logs[date_str].append((kst_time.strftime("%H:%M"), h['role'], h['content'], h.get('correction')))
            
            if not grouped_logs:
                st.info("아직 대화 기록이 없습니다.")
            else:
                for date_str, logs in grouped_logs.items():
                    with st.expander(f"📅 {date_str} (기록 {len(logs)}건)"):
                        for time_str, role, content, correction in reversed(logs):
                            st.caption(f"{time_str} - {role.upper()}")
                            st.write(content)
                            if correction:
                                st.markdown(f'<div class="correction-box" style="margin-bottom:15px;">💡 {correction}</div>', unsafe_allow_html=True)
                            else:
                                st.write("---")


        # -----------------------------------------------------
        # TAB 4: 마이페이지 (개인 맞춤화 학습 커스텀)
        # -----------------------------------------------------
        with tab4:
            st.subheader("⚙️ 마이페이지 및 설정")
            with st.form("settings_form"):
                new_lang = st.selectbox("학습 언어", ["영어", "일본어", "중국어", "한국어"], index=["영어", "일본어", "중국어", "한국어"].index(settings['target_language']))
                new_sit = st.selectbox("상황 설정", ["일상 대화", "비즈니스 미팅", "해외 여행", "공항 입국심사", "카페 주문"], index=["일상 대화", "비즈니스 미팅", "해외 여행", "공항 입국심사", "카페 주문"].index(settings['situation']))
                new_gen = st.radio("선생님 성별", ["여성", "남성"], index=0 if settings['partner_gender'] == '여성' else 1)
                new_level = st.slider("진도 단계 (Level)", 1, 5, value=int(settings['level']))
                
                if st.form_submit_button("설정 저장 및 수정"):
                    supabase.table("users").update({
                        "target_language": new_lang, "situation": new_sit, "partner_gender": new_gen, "current_level": new_level
                    }).eq("id", st.session_state.user_id).execute()
                    st.success("학습 환경 설정이 업데이트되었습니다!")
                    st.rerun()