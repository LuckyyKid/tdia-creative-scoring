import numpy as np
from math import exp
import json

# ==============================================================================
# 1. CONFIGURATION DU CERVEAU (POIDS, BIAIS et DICTIONNAIRES)
# ==============================================================================

# NOTE IMPORTANTE : Les poids F29 et F30 sont initialisés à 0.0. 
# ILS DOIVENT ÊTRE MIS À JOUR avec les valeurs obtenues après votre prochain 
# entraînement ML pour que l'auto-ajustement fonctionne.

# POIDS CALCULÉS PAR LA RÉGRESSION LOGISTIQUE (F1 à F28 sont des exemples)
API_WEIGHTS = {
    'F1': 0.54, 'F2': 0.8262, 'F3': -0.6153, 'F4': 0.0, 'F5': -0.2053, 
    'F6': 0.2014, 'F7': -0.5578, 'F8': 0.7572, 'F9': 0.2358, 'F10': 0.2109, 
    'F11': 0.4218, 'F12': 0.0027, 'F13': 0.0775, 'F14': -0.0535, 'F15': -1.3572, 
    'F16': 0.0, 'F17': 0.023, 'F18': -0.3199, 'F19': 0.0, 'F20': 0.0032, 
    'F21': 0.0, 'F22': 0.1579, 'F23': -0.2343, 'F24': -1.0468, 'F25': 0.41, 
    'F26': -0.6429, 'F27': 0.0, 'F28': 0.0,
    
    # NOUVELLES FEATURES AUTO-AJUSTABLES (Poids initial à 0.0)
    'F29_PainPointMatch': 0.0, # Doit être appris par ML
    'F30_IncentiveMatch': 0.0  # Doit être appris par ML
}
API_BIAS = 0.2109 

# Liste complète des noms de features (30 features)
FEATURE_NAMES = [f'F{i}' for i in range(1, 29)] + ['F29_PainPointMatch', 'F30_IncentiveMatch']

# Mots-clés des points de douleur/bénéfices (Awareness/Consideration)
PAIN_POINTS_BY_INDUSTRY = {
    "underwear": ["inconfort", "irrite", "invisible", "mauvaise qualité", "sensation", "pop underwear", "ride moi"],
    "fashion": ["taille", "coupe", "style", "durabilité", "dernier cri", "tendance", "vieux", "démodé"],
    "beauty": ["rides", "imperfections", "sécheresse", "âge", "éclat", "boutons", "acné", "hydratation", "cernes"],
    "fitness": ["fatigue", "douleur", "stagnation", "perte de poids", "motivation", "résultats", "calories", "régime"],
    "tech": ["lent", "bug", "complexe", "sécurité", "obsolète", "mise à jour", "batterie", "wifi", "piraté"]
}

# Mots-clés d'Incitation à l'Achat (Conversion)
INCENTIVE_KEYWORDS = ["réduction", "rabais", "offre", "promo", "soldes", "gratuit", "free", "%", "off", "deal", "code", "coupon", "exclusif", "achetez", "magasinez", "shop", "économiser", "maintenant", "dernier chance"]


# ==============================================================================
# 2. FONCTIONS HELPER
# ==============================================================================

def map_value(value, mapping, default=0):
    return mapping.get(value, default)

def one_hot(value, categories):
    return [1 if value == c else 0 for c in categories]


# ==============================================================================
# 3. FONCTION D'EXTRACTION DE FEATURES (F1 à F30)
# ==============================================================================

def extract_features(vision):
    """
    Extrait les 30 features (visuelles et contextuelles) du JSON de vision.
    """
    # --- PRÉPARATION DES DONNÉES ---
    
    # Lecture des données contextuelles du JSON unifié
    try:
        industry = vision["external_context"]["industry"].lower().strip()
        # Note: goal est inclus pour la robustesse, mais n'est pas utilisé directement pour F29/F30 binaires
        # goal = vision["external_context"]["goal"].lower().strip() 
    except KeyError:
        # Valeur par défaut en cas d'erreur de JSON pour éviter un crash
        industry = "unknown" 
        
    visible_words = [w.lower() for w in vision["text_elements"]["visible_words"]]
    visible_words_joined = " ".join(visible_words)

    # --- F1 à F28 (Logique existante) ---
    
    # Layout / Hook (F1-F4)
    primary_focus = 1 if vision["layout"]["primary_focus"] == "product" else 0
    framing = one_hot(vision["layout"]["framing"], ["close-up", "medium", "far"])
    product_size_ratio = vision["layout"].get("product_size_ratio", 0)

    contrast_map = {"low": 0, "medium": 1, "high": 2}
    contrast_level = map_value(vision["style"]["contrast_level"], contrast_map)

    brightness_map = {"low": 0, "medium": 1, "high": 2}
    brightness_level = map_value(vision["style"]["brightness_level"], brightness_map)

    # Hold (F7-F12)
    text_blocks = vision["text_elements"]["text_blocks"]
    text_density_map = {"low": 0, "medium": 1, "high": 2}
    text_density = map_value(vision["text_elements"]["text_density"], text_density_map)
    main_headline_present = 1 if vision["text_elements"]["main_headline_present"] else 0

    hierarchy_map = {"unclear": 0, "medium": 1, "clear": 2}
    hierarchy_clarity = map_value(vision["structural_cues"]["hierarchy_clarity"], hierarchy_map)

    visual_noise_map = {"low": 0, "medium": 1, "high": 2}
    visual_noise_level = map_value(vision["structural_cues"]["visual_noise_level"], visual_noise_map)

    whitespace_ratio = vision["structural_cues"].get("whitespace_ratio", 0)

    # CTR (F13-F16)
    cta_present = 1 if vision["text_elements"]["cta_present"] else 0
    
    # Réutilisation des listes de mots-clés
    discount_keywords = ["%", "off", "sale", "ventes flash", "deal"]
    quality_keywords = ["qualité", "bambou", "ultra-doux"]
    guarantee_keywords = ["remboursé", "garantie", "try"]

    has_discount_word = int(any(k in visible_words_joined for k in discount_keywords))
    has_quality_word = int(any(k in visible_words_joined for k in quality_keywords))
    has_guarantee_word = int(any(k in visible_words_joined for k in guarantee_keywords))

    # Style (F17-F21)
    creative_style = vision["style"]["creative_style"]
    creative_style_vector = one_hot(creative_style, ["studio", "ugc", "infographic"])
    ugc_signals = 1 if vision["style"]["ugc_signals"] else 0
    studio_signals = 1 if vision["style"]["studio_signals"] else 0

    # ROAS (F22-F27)
    has_human = 1 if vision["visual_elements"]["has_human"] else 0
    face_visible = 1 if vision["visual_elements"]["face_visible"] else 0

    background_map = {"clean": 0, "medium": 1, "busy": 2}
    background_clarity = map_value(vision["visual_elements"]["background_clarity"], background_map)

    emotion_map = {"smiling": 1, "neutral": 0, "serious": 0}
    emotion_smiling = map_value(vision["emotion_tone"]["emotion_visible"], emotion_map)

    mood_comfortable = int("comfortable" in vision["emotion_tone"]["mood"])

    # Keyword override (F28)
    keyword_pop_ride = 1 if ("pop underwear" in visible_words_joined or "ride moi" in visible_words_joined) else 0

    # --- NOUVELLE LOGIQUE AUTO-AJUSTABLE (F29 & F30) ---
    
    # F29: Pain Point Match (Binaire: 1 si un mot-clé PP est trouvé)
    pain_point_match = 0
    if industry in PAIN_POINTS_BY_INDUSTRY:
        keywords = PAIN_POINTS_BY_INDUSTRY[industry]
        if any(k in visible_words_joined for k in keywords):
            pain_point_match = 1
            
    # F30: Incentive Match (Binaire: 1 si un mot-clé d'incitation à l'achat est trouvé)
    incentive_match = int(any(k in visible_words_joined for k in INCENTIVE_KEYWORDS))
        
    # --- VECTEUR FINAL (F1 à F30) ---
    features = [
        primary_focus
    ] + framing + [
        product_size_ratio,
        contrast_level,
        brightness_level,
        text_blocks,
        text_density,
        main_headline_present,
        hierarchy_clarity,
        visual_noise_level,
        whitespace_ratio,
        cta_present,
        has_discount_word,
        has_quality_word,
        has_guarantee_word
    ] + creative_style_vector + [
        ugc_signals,
        studio_signals,
        has_human,
        face_visible,
        background_clarity,
        emotion_smiling,
        mood_comfortable,
        keyword_pop_ride,          # F28
        pain_point_match,          # F29
        incentive_match            # F30
    ]

    # Vérification critique
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Erreur: Le nombre de features extraites ({len(features)}) ne correspond pas au nombre attendu ({len(FEATURE_NAMES)}).")
        
    return features


# ==============================================================================
# 4. FONCTION DE SCORING PRINCIPALE
# ==============================================================================

def get_creative_score(vision_json_data):
    """
    Calcule le score de performance (probabilité de succès) d'une créative 
    à partir du JSON de Vision complet.
    """
    
    # 1. Extraire le vecteur de features (F1, F2, ..., F30)
    feature_vector = extract_features(vision_json_data) 
        
    # 2. Calculer le Score Brut Z (Z = SUM(Wi * Xi) + Bias)
    Z_score = API_BIAS # Commence par le Biais
    
    for i, feature_value in enumerate(feature_vector):
        feature_name = FEATURE_NAMES[i]
        weight = API_WEIGHTS.get(feature_name, 0.0) # Récupère le poids
        
        Z_score += feature_value * weight
        
    # 3. Convertir Z en Probabilité (Sigmoïde)
    # Probabilité P = 1 / (1 + exp(-Z))
    try:
        probability = 1 / (1 + exp(-Z_score))
    except OverflowError:
        # Gérer le cas où Z_score est trop grand ou trop petit
        probability = 0.0 if Z_score < 0 else 1.0
    
    # 4. Retourner le score
    return {
        "prediction_score": round(probability * 100, 2), # Score en pourcentage
        "Z_score": round(Z_score, 4),
        "context_used": {
            "industry": vision_json_data["external_context"]["industry"],
            "goal": vision_json_data["external_context"]["goal"],
        }
    }