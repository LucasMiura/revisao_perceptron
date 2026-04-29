# Pandas para montar tabelas (DataFrame)
import pandas as pd

# Para salvar e carregar o pipeline de treinamento
import joblib

# Criar pasta/arquivo
from pathlib import Path

# Ferramentas do Sklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class MiniFeatureEngineer:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.model_dir / "mini_transformer.joblib"

        # fit() ou load()
        self.pipeline = None
        self.feature_names = None

    def fit(self, df, num_cols, cat_cols):
        pre = ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )

        self.pipeline = Pipeline(steps=[("preprocessor", pre)])

        # fit = Aprender min/max e categorias
        self.pipeline.fit(df)

        cat = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        self.feature_names = num_cols + cat.get_feature_names_out(cat_cols).tolist()

    def transform(self, df):
        if self.pipeline is None:
            self.load()
        
        # transform = aplica regras aprendidas no fit, sem reaprender
        data = self.pipeline.transform(df)
        return pd.DataFrame(data, columns=self.feature_names)

    def save(self):
        # Salva tudo que foi aprendido no fit
        joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, self.path)

    def load(self):
        state = joblib.load(self.path)
        self.pipeline = state["pipeline"]
        self.feature_names = state["feature_names"]

if __name__ == "__main__":
    # Dados de treino
    df_train = pd.DataFrame([
        {"tempo_entrega": 22, "valor_pedido": 35, "tipo_entrega": "Moto"},
        {"tempo_entrega": 45, "valor_pedido": 90, "tipo_entrega": "Carro"},
        {"tempo_entrega": 30, "valor_pedido": 60, "tipo_entrega": "Bike"}
    ])

    fe = MiniFeatureEngineer()

    # fit aprende as regras do treino
    fe.fit(df_train, ["tempo_entrega", "valor_pedido"], ["tipo_entrega"])

    # Salvar
    fe.save()

    print("Treinamento concluido com sucesso.")