

## Setup

```
# склонируйте репозиторий и перейдите в него
git clone https://github.com/RKhamrin/books-recommendation-system.git
cd books-recommendation-system

# установите зависимости с помощью poetry
poetry install

# активируйте гит-хуки
poetry run pre-commit install
```

## Train

```
poetry run python -m ielts_band_predictor.books_recommendation_system.train
```
