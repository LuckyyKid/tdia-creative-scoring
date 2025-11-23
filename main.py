from fastapi import FastAPI
from pydantic import BaseModel
# Importe la fonction qui contient toute la logique de scoring
from api_scorer import get_creative_score 

app = FastAPI()

# Le modèle de données Pydantic qui représente le JSON complet de Vision
# Il doit inclure toutes les clés, y compris "external_context", pour la validation.
# Nous utilisons 'dict' pour la validation simple, mais vous pourriez la rendre plus stricte.
class VisionData(BaseModel):
    # Nous attendons le JSON complet généré par le prompt GPT Vision
    vision_json: dict 

# Endpoint principal pour calculer le score
@app.post("/score")
def score_creative_endpoint(data: VisionData):
    """
    Reçoit le JSON complet (visuel + contexte), calcule les 30 features
    et retourne le score final de performance.
    """
    try:
        # data.vision_json est le dictionnaire complet qui inclut 
        # "layout", "text_elements", et "external_context".
        score_result = get_creative_score(data.vision_json)
        return score_result
    except KeyError as e:
        # Gère les erreurs si le JSON n'est pas au bon format (ex: missing 'external_context')
        return {"error": f"JSON Vision invalide ou clé manquante: {e}. Vérifiez le format du prompt GPT Vision."}
    except Exception as e:
        return {"error": f"Erreur interne lors du scoring: {e}"}


@app.get("/")
def root():
    return {"message": "TDIA Creative OS API is running 🚀"}

# Votre API est maintenant prête à être déployée sur Render.