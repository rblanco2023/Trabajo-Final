import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle
from joblib import load

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix

filePath = "C:/Users/raul.blanco/Documents/4 - Personales/UBA/Repositorios/IA/Trabajo-Final/test_data.csv"
test_data = pd.read_csv(filepath_or_buffer=filePath, header=0, sep=",")
test_timedelta = test_data[['timedelta']]

filePath = "C:/Users/raul.blanco/Documents/4 - Personales/UBA/Repositorios/IA/Trabajo-Final/y_full.csv"
y_full = pd.read_csv(filepath_or_buffer=filePath, header=0, sep=",")
y = y_full['class']
y_full = y_full.to_numpy()
test_data = test_data.drop(['Unnamed: 0',"timedelta"], axis=1)
test_data = test_data.to_numpy()

select = int(input("Presione 1 para LSTM Autoencoder, 2 para Restricted Boltzmann Machine, 3 para Isolation Forest, 4 para Support Vector Machines: "))

if select == 1:

    model = load('model_autoencoder.joblib')
    test_data_LSTM = test_data.reshape(test_data.shape[0],1,test_data.shape[1])
    predictions_LSTM = model.predict(test_data_LSTM)

    threshold_LSTM = float(input("Modelo LSTM Autoencoder. Ingrese el Threshold a utilizar: "))

    predictions_LSTM[predictions_LSTM >= threshold_LSTM] = 1
    predictions_LSTM[predictions_LSTM < threshold_LSTM] = 0
    predictions_LSTM = predictions_LSTM.reshape(predictions_LSTM.shape[0], predictions_LSTM.shape[2])

    test_data = test_data.reshape(test_data_LSTM.shape[0], test_data_LSTM.shape[2])
    confusion_matrix = pd.crosstab(y_full[:, -1], predictions_LSTM[:, 10], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    pd_predictions_LSTM = pd.DataFrame(predictions_LSTM)

    column_10 = predictions_LSTM[:, 10]
    is_one_LSTM = column_10 ==1

    # Asignar 1 a los elementos que sean True y 0 a los elementos que sean False
    has_ones_LSTM = np.where(is_one_LSTM, 1, 0)

    has_ones_LSTM_pd = pd.DataFrame(has_ones_LSTM, columns=['class'])
    count = (has_ones_LSTM_pd['class'] == 1).sum()
    count1 = (has_ones_LSTM_pd['class'] == 0).sum()
    print(f'El número de anomalias es {count} y {count1} son valores normales')

    new_column_LSTM = has_ones_LSTM

    # Agregar la nueva columna al final del ndarray original
    predictions_class = np.hstack((predictions_LSTM, new_column_LSTM.reshape(-1, 1)))

    pd_predictions_LSTM = pd.DataFrame(predictions_class)

    pd_test_data = pd.DataFrame(test_data)

    print(classification_report(predictions_class[:,14], y))

    y_pd = pd.DataFrame(y, columns = ['class'])

predictions_class_pd  = pd.DataFrame(predictions_class, columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','class'])

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix
class visualization:
  labels = ["Normal","Anomal"]
  def draw_confusion_matrix(self, y, ypred):
    matrix = confusion_matrix(y, ypred)

    plt.figure(figsize=(10,8))
    colors=["orange", "green"]
    sns.heatmap(matrix, xticklabels=self.labels, yticklabels=self.labels, cmap=colors, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("Realidad")
    plt.xlabel("Predicción")
    plt.savefig("confusion_matrix.jpg", dpi=300, bbox_inches='tight')
    #plt.show()

viz = visualization()
viz.draw_confusion_matrix(y_pd, predictions_class[:,14])

from sklearn.metrics import roc_auc_score
clase = y_full[:,-1].astype(int)
# Calcular el AUC para cada modelo utilizando la función roc_auc_score() de la biblioteca scikit-learn
LSTM_auc = roc_auc_score(clase, predictions_class[:,14])

test_timedelta.reset_index(inplace=True)

predictions_class_LSTM_pd = pd.concat([test_timedelta, predictions_class_pd], axis = 1 , join = "inner")

anomalies_LSTM = predictions_class_LSTM_pd.loc[predictions_class_LSTM_pd['class'] == 1.0,['timedelta']]

test_data_pd = pd.DataFrame(test_data, columns=['bx_gse', 'by_gse', 'bz_gse', 'theta_gse',
       'phi_gse', 'bx_gsm', 'by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm', 'bt',
       'density', 'speed', 'temperature'])
test_data_pd = pd.concat([test_timedelta, test_data_pd], axis = 1 , join = "inner")

output = test_data_pd[test_data_pd['timedelta'].isin(anomalies_LSTM['timedelta'])]
output = output.round(3)

output = output.drop(['index'], axis=1)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle

pdf = canvas.Canvas("Reporte de Anomalías.pdf", pagesize=A4)
page_width, page_height = A4
inch = 72  #32
df = output

data = [df.columns.tolist()] + df.values.tolist()

styles = getSampleStyleSheet()
style_table = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 6),
    ('BOTTOMPADDING', (0, 0), (-1, 0),12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 6),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
])

num_rows = len(data)
num_cols = len(data[0])

max_rows_per_page = int((page_height - 5.5 * inch) / (12 * 1.2)) - 1

num_pages = int(num_rows / max_rows_per_page) + 1

for page in range(num_pages):
    # Título en la primera página
    if page == 0:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(inch, page_height - inch, "Reporte de anomalías detectadas")
        pdf.drawString(inch, page_height - inch+18, "INVAP")

    # Agrego la matriz de confusión
        pdf.setFont("Helvetica", 11)
        pdf.drawString(inch, page_height - inch-20, "La matriz de confusión obtenida con el modelo seleccionado es la siguiente:")
        pdf.drawImage("confusion_matrix.jpg", 100, 360, width=5*inch, height=5*inch)
        #pdf.drawImage("confusion_matrix.jpg", 100, 400, width=350, height=350) #, width=3*inch, height=3*inch)

    # Agrego resultados de AUC
        pdf.drawString(inch, 300, f"AUC de Restricted Bolztmann Machine: {LSTM_auc}")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(inch, inch+24, "Informe de anormalidades detectadas")
        pdf.drawString(inch, inch+12, "Especialización en Inteligencia Artificial")
        pdf.drawString(inch, inch, "Ing. Raúl Blanco Elicabe")
        pdf.drawString(inch+420, inch, f"Página {page+1}")
        pdf.showPage()
    if page > 0:
    # Pie de página en todas las páginas
        pdf.setFont("Helvetica", 8)
        pdf.drawString(inch, inch+24, "Informe de anormalidades detectadas")
        pdf.drawString(inch, inch+12, "Especialización en Inteligencia Artificial")
        pdf.drawString(inch, inch, "Ing. Raúl Blanco Elicabe")
        pdf.drawString(inch+420, inch, f"Página {page+1}")


        start_row = (page-1) * max_rows_per_page
        end_row = min((page) * max_rows_per_page, num_rows)

        page_data = data[start_row:end_row]

        table = Table(page_data,colWidths=[0.75 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch])
        
        table.setStyle(style_table)

        table.wrapOn(pdf, page_width - 2 * inch, page_height - 2 * inch)
        table.drawOn(pdf, inch, page_height - 10 * inch - 12 * 1.2)

        if page < num_pages - 1:
            pdf.showPage()

pdf.save()

print("Se generó el archivo <Reporte de Anomalías.pdf> con los resultados obtenidos")