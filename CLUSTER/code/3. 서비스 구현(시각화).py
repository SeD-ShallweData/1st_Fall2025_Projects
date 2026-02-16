# ==================================================================
# 1. 라이브러리 임포트
# ==================================================================
import pandas as pd
import re
import os
from tqdm import tqdm
from transformers import pipeline
from keybert import KeyBERT
from pyvis.network import Network
from sentence_transformers import util, SentenceTransformer
from kiwipiepy import Kiwi

# tqdm 진행률 표시를 pandas에 적용
tqdm.pandas()

# ==================================================================
# 2. AI 모델 및 도구 로드 (서버 시작 시 1회만 실행)
# ==================================================================
print("--- AI 모델 로딩 시작 (서버 시작 시 1회만 실행됩니다) ---")

try:
    summarizer = pipeline("summarization", model="gogamza/kobart-summarization", device=-1)
    print(">>> AI 요약 모델 로딩 완료.")
except Exception as e:
    summarizer = None
    print(f"AI 요약 모델 로딩 실패: {e}")

try:
    sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask', device='cpu')
    print(">>> AI 임베딩 모델(SBERT) 로딩 완료.")
    kw_model = KeyBERT(sbert_model)
    print(">>> AI 키워드 추출 모델(KeyBERT) 로딩 완료.")
except Exception as e:
    kw_model, sbert_model = None, None
    print(f"AI 모델 로딩 실패: {e}")

# 'kiwi'와 'STOPWORDS'는 이제 키워드 추출에 직접 사용되지 않지만, 다른 기능을 위해 남겨둘 수 있습니다.
kiwi = Kiwi()
STOPWORDS = set(["은","는","이","가","을","를","과","와","도","만","로","으로","에","에서","하다","했다","한다","되다","됐다","된다","위해","통해","따라","대한","관련","기자","뉴스","사진","지난","올해","이번","최근","때문","정도","부분","문제"])


# ==================================================================
# 3. 분석 보조 함수 (키워드 추출, 요약, 시각화)
# ==================================================================

def clean_text(text):
    """간단한 텍스트 정제 함수"""
    text = str(text).lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^가-힣a-z0-9\s.,?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_narrative_summary(text_chunk):
    """AI 모델을 사용해 텍스트 요약을 생성합니다."""
    if not summarizer: return "AI 요약 모델이 준비되지 않았습니다."
    try:
        summary = summarizer(text_chunk[:1024], max_length=150, min_length=40, truncation=True)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"   - 요약 생성 중 오류: {e}")
        return "요약 생성 중 오류가 발생했습니다."

def create_keyword_visualization(topic_id, topic_full_text):
    """토픽의 전체 텍스트로부터 키워드를 추출하고 네트워크 시각화 파일을 생성합니다."""
    if not kw_model:
        return [], None

    # ★★★ 변경된 부분 시작 ★★★
    # 기존의 명사만 추출하던 엄격한 방식 대신, KeyBERT가 직접 전체 텍스트에서 키워드를 찾도록 변경합니다.
    # 이렇게 하면 더 유연하고 폭넓은 키워드 추출이 가능합니다.
    try:
        keywords = kw_model.extract_keywords(topic_full_text,
                                             keyphrase_ngram_range=(1, 2), # 1~2개 단어로 이루어진 키워드(구)를 찾습니다.
                                             stop_words=list(STOPWORDS), # 불용어는 여전히 제거합니다.
                                             top_n=10,
                                             use_mmr=True,
                                             diversity=0.7)
        final_keywords = [{"term": term, "score": float(score)} for term, score in keywords if term]
    except Exception as e:
        print(f"   - 키워드 추출 중 오류: {e}")
        return [], None
    # ★★★ 변경된 부분 끝 ★★★
    
    # 네트워크를 만들려면 최소 2개의 키워드가 필요합니다.
    if len(final_keywords) < 2:
        return final_keywords, None # 키워드는 반환하되, 그래프 URL은 반환하지 않습니다.

    net = Network(height="600px", width="100%", notebook=False, cdn_resources='remote', bgcolor="#222222", font_color="white")
    
    for kw in final_keywords:
        net.add_node(kw['term'], label=kw['term'], value=kw['score'] * 100, title=f"Score: {kw['score']:.2f}")

    keyword_terms = [kw['term'] for kw in final_keywords]
    embeddings = sbert_model.encode(keyword_terms, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

    SIMILARITY_THRESHOLD = 0.1
    for i in range(len(keyword_terms)):
        for j in range(i + 1, len(keyword_terms)):
            if cosine_scores[i, j] > SIMILARITY_THRESHOLD:
                net.add_edge(keyword_terms[i], keyword_terms[j], value=float(cosine_scores[i, j]))

    output_dir = os.path.join('static', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    graph_filename = f'network_topic_{topic_id}.html'
    full_path = os.path.join(output_dir, graph_filename)
    net.save_graph(full_path)
    
    graph_url = f"/static/visualizations/{graph_filename}"
    return final_keywords, graph_url


# ==================================================================
# 4. 메인 분석 파이프라인 (단순화된 버전)
# ==================================================================
def run_simplified_analysis(uploaded_file):
    """
    업로드된 Excel 파일을 받아 토픽별 요약 및 키워드 분석을 수행합니다.
    """
    print("\n>>> 단순화된 분석 파이프라인 시작")
    
    try:
        df = pd.read_excel(uploaded_file)
        
        topic_col = 'topic_id'
        text_col = 'topic_text'
        
        if topic_col not in df.columns or text_col not in df.columns:
            raise ValueError("업로드된 Excel 파일에 'topic_id'와 'topic_text' 컬럼이 반드시 포함되어야 합니다.")
            
    except Exception as e:
        raise ValueError(f"파일을 읽는 중 오류 발생: {e}. XLSX 파일 형식과 컬럼명을 확인해주세요.")

    analysis_results = []
    
    print("   - 토픽별 분석 시작 (데이터 행별 직접 처리)")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="토픽 분석 진행률"):
        
        topic_id = row[topic_col]
        topic_full_text = row[text_col]
        
        if int(topic_id) == -1 or not topic_full_text:
            continue
        
        cleaned_text = clean_text(topic_full_text)
        summary = generate_narrative_summary(cleaned_text)
        keywords, graph_url = create_keyword_visualization(topic_id, cleaned_text)
        
        analysis_results.append({
            "id": int(topic_id),
            "title": f"토픽 #{topic_id} 분석 결과",
            "summary": summary,
            "keywords": keywords,
            "graph_url": graph_url,
        })

    print(">>> 모든 분석 완료.")
    return analysis_results