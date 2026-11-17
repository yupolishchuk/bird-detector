## Встановлення залежностей

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Запустити детекцію на відео: 
```
python predict_video.py --input examples/input.mp4 --output examples/output.mp4

# Запустити зі стандартними вагами  для порівняння 
python predict_video.py --input examples/input.mp4 --output examples/output_default.mp4 --weights weights/yolo26n_default.pt

# Передати threshold  
--threshold 0.5
```


## Для запуску навчання:
```
# Запустити навчання (це перезапише поточні ваги моделі weigths/best.ph)
source .venv/bin/activate
python train.py

# можна передати необов'язковий праметр
--epochs 1
```



