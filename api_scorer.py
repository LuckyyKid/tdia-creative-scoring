import numpy as np
from math import exp
import json

# ==============================================================================
# 1. CONFIGURATION DU CERVEAU (POIDS, BIAIS et DICTIONNAIRES)
# ==============================================================================

# POIDS CALCULÉS PAR LA RÉGRESSION LOGISTIQUE (MIS À JOUR AVEC VOS RÉSULTATS)
API_WEIGHTS = {
    'F1': 0.5637, 'F2': 0.846, 'F3': -0.623, 'F4': 0.0, 'F5': -0.1716,
    'F6': 0.1821, 'F7': -0.4521, 'F8': 0.7784, 'F9': 0.2412, 'F10': 0.223,
    'F11': 0.4461, 'F12': 0.0031, 'F13': 0.0825, 'F14': 0.0484, 'F15': -1.0097,
    'F16': 0.0154, 'F17': 0.0, 'F18': -0.3305, 'F19': 0.0, 'F20': 0.0023,
    'F21': 0.0, 'F22': 0.1617, 'F23': -0.2483, 'F24': -0.9596, 'F25': 0.3546,
    'F26': -0.6175, 'F27': 0.0, 'F28': 0.0, # F27 et F28 étaient à 0.0, F17 est à 0.0
    
    # FEATURES AUTO-AJUSTABLES (Poids appris)
    'F29_PainPointMatch': 0.0508, 
    'F30_IncentiveMatch': -0.9251 
}
API_BIAS = 0.223 

# Liste complète des noms de features (30 features)
FEATURE_NAMES = [f'F{i}' for i in range(1, 29)] + ['F29_PainPointMatch', 'F30_IncentiveMatch']

# Mappage des 30 features aux 4 composants et leurs scores max.
# Ces scores sont dérivés du modèle ML par pondération et normalisation.
COMPONENT_MAPPING = {
    # F1:Focus, F2-F4:Framing, F5:Ratio, F6:Contrast, F7:Brightness, F10:Hierarchy, F11:VisualNoise
    "Hook": {"features": ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F10', 'F11'], "max_score": 30},
    # F8:TextBlocks, F9:TextDensity, F12:Whitespace, F20:Studio/UGC-related (Studio/UGC are split but F20 is Studio style)
    "Hold": {"features": ['F8', 'F9', 'F12', 'F20'], "max_score": 20},
    # F13:CTA, F14:Discount, F16:Guarantee, F30:Incentive Match
    "CTR": {"features": ['F13', 'F14', 'F16', 'F30'], "max_score": 25},
    # F15:QualityWord, F17-F19:Style(3), F21:UGC Signals, F22:Studio Signals, F23:Has Human, F24:Face Visible, F25:Background, F26:Emotion Smiling, F27:Mood, F28:Keyword, F29:PainPoint
    "ROAS": {"features": ['F15', 'F17', 'F18', 'F19', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29'], "max_score": 20},
}
# La somme des features dans chaque groupe couvre exactement les 30 FEATURES_NAMES.

# Mots-clés des points de douleur/bénéfices (Awareness/Consideration) - Bilingue
PAIN_POINTS_BY_INDUSTRY = {
    "underwear": ["inconfort", "irrite", "invisible", "mauvaise qualité", "sensation", "pop underwear", "ride moi", "discomfort", "itch", "irritation", "poor quality", "sensation", "invisible", "wedgie"],
    "fashion": ["taille", "coupe", "style", "durabilité", "dernier cri", "tendance", "vieux", "démodé", "size", "fit", "style", "durability", "trend", "old", "outdated", "cheap"],
    "beauty": ["rides", "imperfections", "sécheresse", "âge", "éclat", "boutons", "acné", "hydratation", "cernes", "wrinkles", "blemishes", "dryness", "age", "glow", "acne", "hydration", "dark circles", "pores"],
    "fitness": ["fatigue", "douleur", "stagnation", "perte de poids", "motivation", "résultats", "calories", "régime", "fatigue", "pain", "stuck", "weight loss", "motivation", "results", "calories", "diet", "workout"],
    "tech": ["lent", "bug", "complexe", "sécurité", "obsolète", "mise à jour", "batterie", "wifi", "piraté", "slow", "bug", "complex", "security", "outdated", "update", "battery", "wifi", "hacked", "crash"]
}

# Mots-clés d'Incitation à l'Achat (Conversion) - Bilingue
INCENTIVE_KEYWORDS = [
    "réduction", "rabais", "offre", "promo", "soldes", "gratuit", "free", "%", "off", 
    "deal", "code", "coupon", "exclusif", "achetez", "magasinez", "shop", 
    "économiser", "maintenant", "dernier chance",
    "discount", "sale", "offer", "exclusive", "buy now", "save", "limited time", 
    "flash sale", "bogo", "clearance", "ships free", "coupon", "today" 
]

# ==============================================================================
# 2. FONCTIONS HELPER
# ==============================================================================

def map_value(value, mapping, default=0):
    return mapping.get(value, default)

def one_hot(value, categories):
    return [1 if value == c else 0 for c in categories]

def get_tag_label(score):
    """
    Applique la logique de Tagging (To Kill, To Iterate, To Scale) basée sur le score (0-100).
    """
    if score >= 70:
        return "to scale"
    elif score >= 60:
        return "to iterate"
    else:
        return "to kill"

# ==============================================================================
# 3. FONCTION D'EXTRACTION DE FEATURES (F1 à F30)
# ==============================================================================

def extract_features(vision):
    """
    Extrait les 30 features (visuelles et contextuelles) du JSON de vision.
    Retourne les features sous forme de dictionnaire {nom_feature: valeur}.
    """
    
    # --- PRÉPARATION DES DONNÉES ---
    try:
        industry = vision["external_context"]["industry"].lower().strip()
    except KeyError:
        industry = "unknown" 
        
    visible_words = [w.lower() for w in vision["text_elements"]["visible_words"]]
    visible_words_joined = " ".join(visible_words)

    # --- CALCUL F1 à F28 ---
    # F1: primary_focus
    primary_focus = 1 if vision["layout"]["primary_focus"] == "product" else 0
    # F2-F4: framing
    framing = one_hot(vision["layout"]["framing"], ["close-up", "medium", "far"])
    # F5: product_size_ratio
    product_size_ratio = vision["layout"].get("product_size_ratio", 0)

    # F6: contrast_level
    contrast_map = {"low": 0, "medium": 1, "high": 2}
    contrast_level = map_value(vision["style"]["contrast_level"], contrast_map)

    # F7: brightness_level
    brightness_map = {"low": 0, "medium": 1, "high": 2}
    brightness_level = map_value(vision["style"]["brightness_level"], brightness_map)

    # F8: text_blocks
    text_blocks = vision["text_elements"]["text_blocks"]
    # F9: text_density
    text_density_map = {"low": 0, "medium": 1, "high": 2}
    text_density = map_value(vision["text_elements"]["text_density"], text_density_map)
    # F10: main_headline_present
    main_headline_present = 1 if vision["text_elements"]["main_headline_present"] else 0

    # F11: hierarchy_clarity
    hierarchy_map = {"unclear": 0, "medium": 1, "clear": 2}
    hierarchy_clarity = map_value(vision["structural_cues"]["hierarchy_clarity"], hierarchy_map)

    # F12: visual_noise_level
    visual_noise_map = {"low": 0, "medium": 1, "high": 2}
    visual_noise_level = map_value(vision["structural_cues"]["visual_noise_level"], visual_noise_map)

    # F13: whitespace_ratio
    whitespace_ratio = vision["structural_cues"].get("whitespace_ratio", 0)

    # F14: cta_present
    cta_present = 1 if vision["text_elements"]["cta_present"] else 0
    
    # F15: has_discount_word
    discount_keywords = ["%", "off", "sale", "ventes flash", "deal"]
    has_discount_word = int(any(k in visible_words_joined for k in discount_keywords))
    
    # F16: has_quality_word
    quality_keywords = ["qualité", "bambou", "ultra-doux"]
    has_quality_word = int(any(k in visible_words_joined for k in quality_keywords))
    
    # F17: has_guarantee_word
    guarantee_keywords = ["remboursé", "garantie", "try"]
    has_guarantee_word = int(any(k in visible_words_joined for k in guarantee_keywords))

    # F18-F20: creative_style
    creative_style = vision["style"]["creative_style"]
    creative_style_vector = one_hot(creative_style, ["studio", "ugc", "infographic"])
    
    # F21: ugc_signals
    ugc_signals = 1 if vision["style"]["ugc_signals"] else 0
    # F22: studio_signals
    studio_signals = 1 if vision["style"]["studio_signals"] else 0

    # F23: has_human
    has_human = 1 if vision["visual_elements"]["has_human"] else 0
    # F24: face_visible
    face_visible = 1 if vision["visual_elements"]["face_visible"] else 0

    # F25: background_clarity
    background_map = {"clean": 0, "medium": 1, "busy": 2}
    background_clarity = map_value(vision["visual_elements"]["background_clarity"], background_map)

    # F26: emotion_smiling
    emotion_map = {"smiling": 1, "neutral": 0, "serious": 0}
    emotion_smiling = map_value(vision["emotion_tone"]["emotion_visible"], emotion_map)

    # F27: mood_comfortable
    mood_comfortable = int("comfortable" in vision["emotion_tone"]["mood"])

    # F28: keyword_pop_ride
    keyword_pop_ride = 1 if ("pop underwear" in visible_words_joined or "ride moi" in visible_words_joined) else 0

    # --- F29 & F30 (Auto-Ajustables) ---
    
    # F29: Pain Point Match (BILINGUE)
    pain_point_match = 0
    if industry in PAIN_POINTS_BY_INDUSTRY:
        keywords = PAIN_POINTS_BY_INDUSTRY[industry]
        if any(k in visible_words_joined for k in keywords):
            pain_point_match = 1
            
    # F30: Incentive Match (BILINGUE)
    incentive_match = int(any(k in visible_words_joined for k in INCENTIVE_KEYWORDS))
        
    # --- Création du dictionnaire final ---
    
    # Flattening features F2-F4 and F18-F20
    framing_features = dict(zip(['F2', 'F3', 'F4'], framing))
    style_features = dict(zip(['F17', 'F18', 'F19'], creative_style_vector))
    
    
    feature_dict = {
        'F1': primary_focus, **framing_features, 'F5': product_size_ratio, 
        'F6': contrast_level, 'F7': brightness_level, 'F8': text_blocks, 
        'F9': text_density, 'F10': main_headline_present, 'F11': hierarchy_clarity,
        'F12': visual_noise_level, 'F13': whitespace_ratio, 'F14': cta_present,
        'F15': has_discount_word, 'F16': has_quality_word, 'F17': has_guarantee_word, # Adjusted indices
        **style_features,
        'F21': ugc_signals, 'F22': studio_signals, 'F23': has_human, 
        'F24': face_visible, 'F25': background_clarity, 'F26': emotion_smiling, 
        'F27': mood_comfortable, 'F28': keyword_pop_ride, 
        'F29_PainPointMatch': pain_point_match, 'F30_IncentiveMatch': incentive_match
    }

    # Rebuilding the feature vector in correct order for simplicity
    feature_vector = [feature_dict.get(name, 0) for name in FEATURE_NAMES]

    if len(feature_vector) != len(FEATURE_NAMES):
        raise ValueError(f"Erreur: Le nombre de features extraites ({len(feature_vector)}) ne correspond pas au nombre attendu ({len(FEATURE_NAMES)}).")
        
    return feature_dict


# ==============================================================================
# 4. FONCTION DE SCORING PRINCIPALE
# ==============================================================================

def calculate_component_scores(feature_dict):
    """Calcule les scores Hook, Hold, CTR, ROAS basés sur le Z-score partiel."""
    
    component_scores = {}
    
    for component, config in COMPONENT_MAPPING.items():
        Z_comp = 0.0
        
        # 1. Calcul du Z-score partiel pour ce composant
        for feature_name in config["features"]:
            feature_value = feature_dict.get(feature_name, 0)
            weight = API_WEIGHTS.get(feature_name, 0.0)
            
            Z_comp += feature_value * weight
            
        # 2. Conversion en Probabilité (Sigmoïde)
        # La probabilité est ici une mesure normalisée de l'efficacité du composant
        try:
            P_comp = 1 / (1 + exp(-Z_comp))
        except OverflowError:
            P_comp = 0.0 if Z_comp < 0 else 1.0
            
        # 3. Mise à l'échelle vers le score max du composant
        scaled_score = round(P_comp * config["max_score"], 2)
        component_scores[component] = scaled_score
        
    return component_scores

def get_creative_score(vision_json_data):
    """
    Calcule le score, le Z_score, le label de tagging et les scores par composant.
    """
    
    # 1. Extraire le dictionnaire de features (F1, F2, ..., F30)
    feature_dict = extract_features(vision_json_data) 
    
    # Reconstruire le vecteur de features dans le bon ordre pour le calcul global
    feature_vector = [feature_dict.get(name, 0) for name in FEATURE_NAMES]
        
    # 2. Calculer le Score Brut Z GLOBAL
    Z_score_global = API_BIAS 
    
    for i, feature_value in enumerate(feature_vector):
        feature_name = FEATURE_NAMES[i]
        weight = API_WEIGHTS.get(feature_name, 0.0) 
        Z_score_global += feature_value * weight
        
    # 3. Conversion en Probabilité et Score Global
    try:
        probability = 1 / (1 + exp(-Z_score_global))
    except OverflowError:
        probability = 0.0 if Z_score_global < 0 else 1.0
    
    final_score = round(probability * 100, 2)
    
    # 4. DÉTERMINER LE LABEL DE TAGGING
    tag_label = get_tag_label(final_score)
    
    # 5. CALCULER LES SCORES PAR COMPOSANT (Hook, Hold, CTR, ROAS)
    component_scores = calculate_component_scores(feature_dict)
    
    # 6. Retourner le résultat complet
    return {
        "prediction_score": final_score, 
        "tag_label": tag_label, 
        "component_scores": component_scores, # Les scores détaillés sont ajoutés ici
        "Z_score": round(Z_score_global, 4),
        "context_used": {
            "industry": vision_json_data["external_context"]["industry"],
            "goal": vision_json_data["external_context"]["goal"],
        }
    }