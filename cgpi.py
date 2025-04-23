# Importa o OpenCV, a biblioteca principal para processar imagens.
import cv2

# Aqui carregamos um modelo pré-treinado que consegue detectar rostos humanos.
# Esse .xml é um arquivo que já vem com o OpenCV (ou pode ser baixado) e contém um
# modelo baseado em Haar features.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Lê uma imagem chamada foto.jpg
img = cv2.imread('foto.png')

# Converte a imagem para escala de cinza. O Haar Cascade funciona melhor com
# imagens em preto e branco, porque ele analisa contraste e não cores
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Essa linha é onde a mágica acontece:
# gray: imagem em escala de cinza.
# 1.1: fator de escala (quanto menor, mais preciso e lento).
# 4: número de vizinhos necessários para confirmar um rosto (quanto maior, mais
# restrito).
# Esse comando retorna uma lista com os rostos encontrados. Cada rosto é uma tupla com
# as coordenadas (x, y, largura, altura).
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Vai percorrer cada rosto detectado e recorta aquela área da imagem original. Essa
# região (ROI – region of interest) pode ser usada depois para verificar se a pessoa está
# usando máscara.
for (x, y, w, h) in faces:
    face_roi = img[y:y+h, x:x+w]
    
    # Desenha um retângulo azul em volta do rosto. Os parâmetros são:
    # (x, y) → canto superior esquerdo
    # (x+w, y+h) → canto inferior direito
    # (255, 0, 0) → cor azul em BGR
    # 2 → espessura da linha
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Exibe a imagem com os rostos detectados. O waitKey(0) faz com que o programa
# espere até você apertar alguma tecla. destroyAllWindows() fecha a janela depois.
cv2.imshow('Resultado', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
