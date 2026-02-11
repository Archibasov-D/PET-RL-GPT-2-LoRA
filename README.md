# PET-RL-GPT-2-LoRA

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

RL-адаптация GPT-2 с LoRA для контролируемой генерации IMDb-отзывов.
# Цель: С помощью LoRA и алгоритма GRPO изменить вероятность генерации GPT-2 положительного отзыва на фильм.

# Датасет 

 Ревью будут генерироваться на основе датасета IMDB. Этот датасет содержит англоязычные отзывы о фильмах с метками: 0 - отрицательный отзыв, 1 - положительный отзыв.
# Базовая модель 

 В качестве базовой модели возьмём `lvwerra/gpt2-imdb`  - GPT-2 fine-tuned на датасете IMDB.

# RL обучение 

## Обучение функции награды

Алгоритм GRPO,   реализованный в библиотеке `trl` из экосистемы Hugging Face,  требует функции награды, которая будет определять направление обучения.

 В качестве функции награды возьмём предобученный BERT `distilbert-base-cased`

 Затем  построим датасет на основе IMDB, который будет содержать пары положительного и отрицательного отзывов. И дообучим  на этом датасете модель награды, чтобы она хорошо различала ревью и давала большую награду положительным отзывам.

Для этого использовался `trl.RewardTrainer`, снизу переведен график роста accuracy модели награды, выше accuracy - чаще правильное ранжирование ревью.

![награда функция.png | 500x400](https://github.com/Archibasov-D/PET-RL-GPT-2-LoRA/blob/full_pipeline/reports/figures/%D0%BD%D0%B0%D0%B3%D1%80%D0%B0%D0%B4%D0%B0%20%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F.png)

## Обучение GPT-2 + LoRA

Для упрощения и ускорения процесса обучения использовалась LoRA поверх предобученного GPT-2.  LoRA -  это небольшая низкоранговая добавка к линейным слоям GPT-2, а следовательно обучение более стабильно и риск сломать модель ниже, так как добавка к базовой модели небольшая. 
Реализация LoRA была взята из библиотеки `peft`,  ранг LoRA равен 32. 

Также для обучения через GRPO потребовалось создать датасет префиксов отзывов длиною в 2-8 токенов.

Графики обучения через trl.GRPOTrainer, логирование велось в wandb:
![LR.png| 500x400](https://github.com/Archibasov-D/PET-RL-GPT-2-LoRA/blob/full_pipeline/reports/figures/LR.png)
![средняя награда.png| 500x400](https://github.com/Archibasov-D/PET-RL-GPT-2-LoRA/blob/full_pipeline/reports/figures/%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D1%8F%D1%8F%20%D0%BD%D0%B0%D0%B3%D1%80%D0%B0%D0%B4%D0%B0.png)

## Влияние learning rate 

Так как trl.GRPOTrainer использует линейный scheduler, то learning rate со временем убывает и скорость этого убывания влияет на характер сходимости, было обнаружено что более медленное уменьшение learning rate приводит к сходимости к большей средней награде, то есть к генерации более позитивных отзывов

Так как эксперимент с большим learning rate обучался дольше, то сравнение не совсем корректное, но видно что синяя кривая отделяется от желтой и идёт выше по размеру средней награды:

![сравнение.png| 500x400](https://github.com/Archibasov-D/PET-RL-GPT-2-LoRA/blob/full_pipeline/reports/figures/%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5.png)

![сравнение LR.png| 500x400](https://github.com/Archibasov-D/PET-RL-GPT-2-LoRA/blob/full_pipeline/reports/figures/%D1%81%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20LR.png)

# Сравнение базовой и RL модели

Для анализа влияния RL на вероятность генерации положительного отзыва через GPT-2, будем использовать большую модель RoBERTA: `siebert/sentiment-roberta-large-english`

На основе IMDB был создан датасет из 1500 префиксов ревью, длиною в 2-8 токенов, аналогично процедуре обучения через GRPO.
Затем эти ревью были независимо дописаны двумя моделями, а затем семантически оценены, процент положительных отзывов представлен в таблице:

| Модель       | Процент положительных отзывов |
| ------------ | ----------------------------- |
| GPT-2        | 49.6%                         |
| GPT-2 + LoRA | 57.1%                         |


Видно что в результате RL обучения модель значительно чаще начала генерировать положительные отзывы.

# Дальнейшие идеи

- Эксперименты с learning rate для поиска оптимального обучения
- Эксперименты с рангом LoRA






## Project Organization 

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

