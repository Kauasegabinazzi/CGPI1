# Importa o OpenCV, a biblioteca principal para processar imagens.
import cv2

# Aqui carregamos um modelo pré-treinado que consegue detectar rostos humanos.
# Esse .xml é um arquivo que já vem com o OpenCV (ou pode ser baixado) e contém um modelo baseado em Haar features.
# Carrega os classificadores Haar pré-treinados

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_cascade = cv2.CascadeClassifier('haarcascade_mask.xml')  # Detecta máscara

# Verifica se os arquivos foram carregados corretamente
if face_cascade.empty():
    print("Erro ao carregar haarcascade_frontalface_default.xml")
if mask_cascade.empty():
    print("Erro ao carregar haarcascade_mask.xml")

# Lê uma imagem
img = cv2.imread('foto12.png')

# Verifica se a imagem foi carregada corretamente
if img is None:
    print("Erro ao carregar a foto.")
    exit()

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

# Detecta máscaras na imagem
masks = mask_cascade.detectMultiScale(gray, 1.04, 3)

# Vai percorrer cada rosto detectado e recorta aquela área da imagem original
# Para cada máscara detectada, desenha um retângulo verde
for (x, y, w, h) in faces:
    # Região de interesse (somente o rosto)

    face_roi = img[y:y+h, x:x+w]
    # Desenha um retângulo azul em volta do rosto. Os parâmetros são:
    # # (x, y) → canto superior esquerdo
    # # (x+w, y+h) → canto inferior direito
    # # (255, 0, 0) → cor azul em BGR
    # # 2 → espessura da linha
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Esse laço percorre todas as máscaras detectadas na imagem
    # Cada item da lista é uma tupla com 4 valores:
    # (mx, my, mw, mh)
    # mx → coordenada X do canto superior esquerdo
    # my → coordenada Y do canto superior esquerdo
    # mw → largura da área
    # mh → altura da área

    for (mx, my, mw, mh) in masks:
        if len(masks) > 0:
            # Desenha um retângulo verde em volta da máscara detectada. Os parâmetros são:
            # (mx, my) → canto superior esquerdo
            # (mx+mw, my+mh) → canto inferior direito
            # (0, 255, 0) → cor verde em BGR (indica máscara encontrada)
            # 2 → espessura da linha
            cv2.rectangle(img, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)  # Verde

# Exibe a imagem com os rostos detectados. O waitKey(0) faz com que o programa
# espere até você apertar alguma tecla. destroyAllWindows() fecha a janela depois.
cv2.imshow('Resultado', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
