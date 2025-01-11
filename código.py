import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Dataset de exemplo: Sentimento em textos (positivo ou negativo)
data = {
    'textos': [
        "Eu amo este produto", "Este é o pior serviço", "Muito bom e eficiente", 
        "Horrível, nunca mais compro", "Excelente qualidade", "Não gostei nada disso", 
        "Maravilhoso, recomendo!", "Terrível, um desastre", "Muito satisfeito", "Que decepção!",
        "Produto maravilhoso", "Experiência péssima", "Gostei muito do atendimento",
        "Horrível, não recomendo", "Qualidade incrível", "O atendimento deixou a desejar",
        "Muito ruim", "Fantástico, superou as expectativas", "Jamais comprarei novamente",
        "Produto excelente, adorei"
    ],
    'sentimento': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  # 1 = Positivo, 0 = Negativo
}

# Convertendo para DataFrame
df = pd.DataFrame(data)

# Pré-processamento dos dados
X = df['textos']
y = df['sentimento']

# Comparando dois métodos de vetorização: Bag of Words e TF-IDF
vectorizer_bow = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()

X_bow = vectorizer_bow.fit_transform(X)
X_tfidf = vectorizer_tfidf.fit_transform(X)

# Divisão dos dados em treino e teste para ambos os métodos
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.3, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Treinamento com Naive Bayes usando Bag of Words
modelo_nb_bow = MultinomialNB()
modelo_nb_bow.fit(X_train_bow, y_train)

# Treinamento com Naive Bayes usando TF-IDF
modelo_nb_tfidf = MultinomialNB()
modelo_nb_tfidf.fit(X_train_tfidf, y_train)

# Treinamento com Random Forest usando Bag of Words
modelo_rf_bow = RandomForestClassifier(random_state=42)
modelo_rf_bow.fit(X_train_bow, y_train)

# Predições
y_pred_nb_bow = modelo_nb_bow.predict(X_test_bow)
y_pred_nb_tfidf = modelo_nb_tfidf.predict(X_test_tfidf)
y_pred_rf_bow = modelo_rf_bow.predict(X_test_bow)

# Avaliação do modelo
print("Naive Bayes com Bag of Words")
print("Acurácia:", accuracy_score(y_test, y_pred_nb_bow))
print(classification_report(y_test, y_pred_nb_bow))
matriz_confusao_nb_bow = confusion_matrix(y_test, y_pred_nb_bow)

print("Naive Bayes com TF-IDF")
print("Acurácia:", accuracy_score(y_test, y_pred_nb_tfidf))
print(classification_report(y_test, y_pred_nb_tfidf))
matriz_confusao_nb_tfidf = confusion_matrix(y_test, y_pred_nb_tfidf)

print("Random Forest com Bag of Words")
print("Acurácia:", accuracy_score(y_test, y_pred_rf_bow))
print(classification_report(y_test, y_pred_rf_bow))
matriz_confusao_rf_bow = confusion_matrix(y_test, y_pred_rf_bow)

# Visualização das Matrizes de Confusão
def plot_confusion_matrix(matriz, title, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

plot_confusion_matrix(matriz_confusao_nb_bow, "Naive Bayes com Bag of Words", ["Negativo", "Positivo"])
plot_confusion_matrix(matriz_confusao_nb_tfidf, "Naive Bayes com TF-IDF", ["Negativo", "Positivo"])
plot_confusion_matrix(matriz_confusao_rf_bow, "Random Forest com Bag of Words", ["Negativo", "Positivo"])
