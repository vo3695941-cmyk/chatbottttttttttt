import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chay_chatbot():
    # 1. Doc du lieu tu file csv da tao tren GitHub
    try:
        df = pd.read_csv('data.csv')
    except Exception as e:
        print("Loi: Khong tim thay file data.csv hoac file bi loi!", e)
        return

    # 2. Chuyen doi van ban thanh cac vector so hoc (TF-IDF)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['question'].values.astype('U'))

    print("=== AI CHATBOT DA SAN SANG (Go 'exit' de thoat) ===")
    
    # 3. Vong lap nhan tin nhan tu nguoi dung
    while True:
        cau_hoi_user = input("Ban nhắn: ")
        if cau_hoi_user.lower() == 'exit':
            print("Tam biet!")
            break
            
        # Tinh toan do tuong dong giua cau hoi moi va du lieu da hoc
        user_vec = vectorizer.transform([cau_hoi_user])
        similarity = cosine_similarity(user_vec, X)
        vi_tri_giong_nhat = similarity.argmax()
        
        # Neu do tuong dong lon hon 20% thi tra loi, nguoc lai bao khong hieu
        if similarity[0][vi_tri_giong_nhat] < 0.2:
            print("AI phan hoi: Xin loi, mo hinh tren GitHub cua ban chua duoc hoc cau nay.")
        else:
            print("AI phan hoi:", df['answer'].iloc[vi_tri_giong_nhat])

if __name__ == "__main__":
    chay_chatbot()
