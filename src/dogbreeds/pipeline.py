"""La pipeline de classification complète."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .features import HOGFeatures, PerClassPCA


def build_pipeline(n_components=5, classifier=None, scale=True, memory=None):
    """MinMax -> (PCA par classe + HOG) -> StandardScaler -> classifieur.

    Chaque étape est apprise sur le train uniquement : rien ne fuit depuis le test.
    `memory` met en cache les étapes de features, ce qui accélère beaucoup les recherches sur grille.
    pipeline[:-1] donne l'extracteur de features sans le classifieur.
    """
    features = FeatureUnion([
        ("pca", PerClassPCA(n_components=n_components)),
        ("hog", HOGFeatures()),
    ])
    steps = [("minmax", MinMaxScaler()), ("all_features", features)]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier if classifier is not None else KNeighborsClassifier(n_neighbors=5)))
    return Pipeline(steps, memory=memory)
