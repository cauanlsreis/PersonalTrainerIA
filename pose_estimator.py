import cv2

file_name = 'IMG_2149.mp4'

cap = cv2.VideoCapture(file_name)

while (cap.isOpened()): #executa enquanto o arquivo do video estiver iterado
    #ret (retorna true ou false para o final do video) frame recolhe os pixels da imagem
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)

        #Aperta Q para fechar o video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release() #encerra a captura de video
cv2.destroyAllWindows() #fecha as janelas criadas pelo OpenCV