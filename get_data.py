import json

def get_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    dataset = []
    for i in data:
        dataset.append(data[i]["thinking"])
    return dataset


if __name__ == '__main__':
    dataset = get_data('/home/user/.cache/kagglehub/datasets/kkhubiev/russian-financial-news/versions/3/RussianFinancialNews/news_descriptions/news_description_LLama3_8b.json')
    print(dataset)