# Coauthor-Recommendation-System
This project uses large language models (LLMs) and network analysis to predict potential research collaborations. It analyzes academic data such as authors and publications to identify meaningful co-author relationships. The system helps researchers discover suitable collaborators through intelligent, data-driven recommendations.

## Features

- Research paper recommendation
- Semantic similarity analysis
- Author co-authorship network analysis
- Research community clustering
- Top-N paper recommendations
- Interactive visualizations

## Technologies Used

- Python
- Flask
- Pandas
- NetworkX
- Scikit-learn
- Sentence Transformers
- all-mpnet-base-v2
- Matplotlib
- Plotly

## How It Works

1. Research paper abstracts are processed.
2. Sentence embeddings are generated using all-mpnet-base-v2.
3. Cosine similarity is calculated between papers.
4. Co-authorship networks are created from author data.
5. Clustering groups similar authors and papers.
6. The system recommends the most relevant papers to the user.

## Installation

```bash
git clone https://github.com/your-username/research-paper-recommendation-system.git
cd research-paper-recommendation-system
pip install -r requirements.txt
```

## Run the Project

```bash
python app.py
```

Open:

```
http://localhost:5000
```

## Project Structure

```
├── app.py
├── recommend.py
├── dataset/
├── templates/
├── static/
├── requirements.txt
└── README.md
```



## Future Improvements

- Citation network analysis
- User personalization
- Real-time paper updates
- Advanced graph analytics

