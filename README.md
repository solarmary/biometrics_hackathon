### Прототип системы обнаружения атак представления
## Постановка задачи
Разработать и продемонстрировать пассивный (не требующий явных действий от пользователя) алгоритм защиты от атак предъявления, способный исполняться на мобильных устройствах. Спектр отсекаемых данным алгоритмом инструментов атак ограничен воспроизведением фото и видео с экранов телефонов, планшетов, ноутбуков, переносных мониторов с видимыми границами устройства воспроизведения и распечатанными фотографиями лиц с видимыми границами фотографии.
## Содержание отправки
* start.py - главный программный файл для обнаружения атаки. Именно его и следуте запускать для проверки работы команды. Запуск осуществляется следуюшим образом: ```python start.py <видеофайл.mp4> ``` где в качестве параметра передается путь до односекундного или больше видеофала, из которого будут взяты три кадра для анализа. По окончанию работы программы будут выведены результаты:
  * атака обнаружена/атака не обнаружена
  * вероятность атаки (порог отсечения >=0.4)
  * время затраченное на анализ
* Файлы detector*.py - программы предикторов кторые задействуются в start.py
* Папки detector* - дополнительные файлы для работы предикторов
* Файл requirements.txt необходимый библиотеки
* demo.ipynb - юпитер ноутбук, удобный для того чтобы вручную поисследовать процесс работы программы. Файл необязательный, для ознакомительных целей
* папки video4test ,  video4test/attack ,  video4test/noattack  - папки для размещения видео, если вы будете работать с demo.ipynb
* папка other - файлы проекта со стадии разработки, в т.ч. собранный датасет
## Проверка работоспособности
Запустите ```python start.py <видеофайл.mp4> ```