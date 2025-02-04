# **Real-Time Product Recommendation System**  

**Project:** A real-time product recommendation system for an e-commerce platform using collaborative filtering and content-based filtering. Implemented as a **Streamlit application**.  

## ** Features**
- **Collaborative Filtering**: Uses KNN-based user similarity for personalized recommendations.  
- **Content-Based Filtering**: Utilizes product attributes and cosine similarity.  
- **Efficient Data Handling**: Optimized memory usage with chunk-based data loading.  
- **Streamlit Web App**: Interactive UI for real-time recommendations.  

## **📂 Project Structure**
```
📦 project-repo
 ┣ 📜 app.py                       # Streamlit app implementation
 ┣ 📜 requirements.txt             # Required dependencies
 ┣ 📜 README.md                    # Documentation
```

## **📊 Dataset**
This project uses the **Instacart Market Basket Analysis** dataset, which contains user purchase history from an online grocery store.  

🔗 **Dataset Link:** [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data)  

## **🛠️ Installation & Usage**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/project-repo.git
cd project-repo
```
### **2️⃣ Set Up Virtual Environment (Recommended)**
```sh
conda create --name recc_system python=3.11
conda activate recc_system
```
### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```
### **4️⃣ Run the Application**
```sh
streamlit run app.py
```
Open the displayed **localhost URL** in your browser to interact with the app.

## **📝 Notes**
- Ensure the **Instacart dataset** is downloaded and placed in the project directory.  
- The application supports **both user-based and product-based recommendations**.  
