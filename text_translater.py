import cv2
import pytesseract
import asyncio
from concurrent.futures import ThreadPoolExecutor
from translate import Translator

# Установка пути к Tesseract OCR (указывайте свой путь)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

async def translate_text(text):
    # Перевод текста на русский
    translator = Translator(to_lang="ru")
    return await asyncio.to_thread(translator.translate, text)

async def process_text_element(frame, text_element):
    x, y, w, h, text = text_element
    try:
        roi = frame[y:y+h, x:x+w]
        translation = await translate_text(text)
        return x, y, w, h, f'{translation}'
    except ValueError:
        print("Ошибка при преобразовании координат")

async def process_frame(frame):
    # Извлечение текста и координат bounding box'а с кадра
    config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(frame, config=config)

    tasks = []
    text_elements = []

    for i, el in enumerate(data.splitlines()):
        el = el.split()
        if len(el) == 12 and el[0].isdigit():
            text_elements.append((int(el[6]), int(el[7]), int(el[8]), int(el[9]), el[11]))

    # Обработка текстовых элементов в параллель
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        tasks = [process_text_element(frame, text_element) for text_element in text_elements]
        translations = await asyncio.gather(*tasks)

    # Обновляем подписи на изображении с переведенными текстами
    for translation in translations:
        if translation is not None:
            x, y, w, h, text = translation
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    # Отображение обработанного кадра
    cv2.imshow('Text Detection', frame)

async def capture_frames(cap):
    while True:
        ret, frame = cap.read()

        # Уменьшение размера кадра перед обработкой
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Обработка кадра в асинхронном режиме
        await process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def main():
    cap = cv2.VideoCapture(1)  # Изменил на 0, чтобы использовать встроенную камеру

    # Установка размера кадра
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Установка количества выводимых кадров в секунду
    cap.set(cv2.CAP_PROP_FPS, 15)

    capture_task = asyncio.create_task(capture_frames(cap))

    await asyncio.gather(capture_task)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
