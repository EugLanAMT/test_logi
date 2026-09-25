
def graph_to_content(graph: FlowchartGraph, title: str = "Logigramme") -> str:
    """Transforme le graphe structuré en une description textuelle en langage
    naturel, destinée à être indexée dans le champ "content" d'Azure AI Search
    (recherche full-text ET recherche vectorielle si vous embeddez ce texte).
 
    Choix volontaire : une traversée DÉTERMINISTE du graphe (pas un nouvel
    appel LLM). Le contenu indexé doit être fidèle à 100% à ce que le graphe
    contient, reproductible, et sans risque d'hallucination ou d'omission
    lors d'une régénération -- ce que ne garantit pas une reformulation libre
    par un modèle.
    """
    nodes_by_id = {n.id: n for n in graph.nodes}
    outgoing: dict = {}
    for e in graph.edges:
        outgoing.setdefault(e.source, []).append(e)
 
    visited = set()
    lines: List[str] = [f"{title} :"]
 
    def describe(node_id: str, depth: int = 0):
        node = nodes_by_id.get(node_id)
        if node_id in visited:
            label = node.text if node else node_id
            lines.append("  " * depth + f"-> Retour à l'étape « {label} » (boucle).")
            return
        if not node:
            return
        visited.add(node_id)
        prefix = "  " * depth
 
        if node.shape_type == "start":
            lines.append(f"{prefix}Début du processus : « {node.text} »")
        elif node.shape_type == "end":
            lines.append(f"{prefix}Fin du processus : « {node.text} »")
        elif node.shape_type == "decision":
            lines.append(f"{prefix}Décision : « {node.text} »")
        else:
            lines.append(f"{prefix}Étape : « {node.text} »")
 
        next_edges = outgoing.get(node_id, [])
        if node.shape_type == "decision":
            for e in next_edges:
                cond = e.label or "cas non précisé"
                lines.append(f"{prefix}  Si {cond} :")
                describe(e.target, depth + 2)
        else:
            for e in next_edges:
                describe(e.target, depth + 1)
 
    # Points de départ : nœuds "start", sinon nœuds sans arête entrante.
    start_nodes = [n for n in graph.nodes if n.shape_type == "start"]
    if not start_nodes:
        incoming_targets = {e.target for e in graph.edges}
        start_nodes = [n for n in graph.nodes if n.id not in incoming_targets] or graph.nodes[:1]
 
    for s in start_nodes:
        describe(s.id)
 
    # Nœuds jamais atteints (composantes déconnectées, erreurs de détection résiduelles).
    unreached = [n for n in graph.nodes if n.id not in visited]
    if unreached:
        lines.append("\nÉtapes non reliées au flux principal détecté :")
        for n in unreached:
            lines.append(f"  - « {n.text} » ({n.shape_type})")
 
    return "\n".join(lines)
 
"""
Amélioration de la détection de logigrammes + robustification de ingest_pdf.

Changement de philosophie :
- L'ancienne is_diagram() jugeait un logigramme sur des seuils géométriques
  absolus (nombre de formes fermées, nombre de lignes...). C'est fragile :
  un logigramme "simple" (6-8 formes) tombe sous les seuils, un logigramme
  scanné en basse qualité change complètement l'edge_ratio, etc.
- La nouvelle approche : la CV ne sert plus qu'à écarter les cas ÉVIDENTS
  (photo quasi unie, texture de photo bruitée) pour économiser des appels.
  La décision réelle "est-ce un logigramme ?" + l'extraction de sa structure
  (nœuds/arêtes) se font en UN SEUL appel à un modèle de vision, qui répond
  en JSON structuré. C'est beaucoup plus robuste sur la variabilité réelle
  des scans/exports PDF, et ça évite de payer deux fois (classification +
  extraction) pour la même image.
- Un fallback CV (seuils assouplis) prend le relais si l'appel au modèle
  échoue, pour ne jamais faire planter toute l'ingestion sur une erreur
  réseau / rate limit.
"""

import re
import json
import numpy as np
import cv2


# ==============================================================================
# 1) PRE-FILTRE CV — bon marché, ne fait QUE écarter les cas évidents
# ==============================================================================

def _quick_visual_signal(image_bytes):
    """
    Ne dit jamais "c'est un logigramme". Dit seulement :
      - False : clairement pas la peine de vérifier (image trop petite,
        quasi unie, ou texture de photo pure)
      - True  : cas plausible ou ambigu -> à faire trancher par le modèle
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return False

    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    total_area = h * w
    edge_ratio = np.sum(edges > 0) / total_area

    # Bornes volontairement larges : on veut juste éliminer les cas
    # sans ambiguïté (page quasi blanche, ou photo texturée façon bruit).
    if edge_ratio < 0.0008 or edge_ratio > 0.35:
        return False

    return True


def _legacy_cv_fallback(image_bytes):
    """
    Filet de sécurité si l'appel au modèle de vision échoue.
    Seuils très assouplis par rapport à l'ancienne version (qui exigeait
    closed_shapes >= 10 et large_shapes >= 5, trop strict pour un
    logigramme simple à 6-8 formes).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    total_area = h * w

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=25, maxLineGap=8
    )
    line_count = len(lines) if lines is not None else 0

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    closed_shapes = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * 0.001:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) >= 4:
            closed_shapes += 1

    # Seuils bas : un logigramme simple a au moins quelques formes et
    # quelques flèches. On préfère un faux positif (repassera par
    # generate_caption si le graphe extrait est vide) plutôt qu'un faux
    # négatif qui fait perdre la page.
    return line_count >= 5 and closed_shapes >= 3


# ==============================================================================
# 2) CLASSIFICATION + EXTRACTION EN UN SEUL APPEL AU MODELE DE VISION
# ==============================================================================

_VISION_PROMPT = """Tu analyses une image extraite d'un document technique \
(procédure, manuel de diagnostic, etc.).

Réponds UNIQUEMENT avec un JSON strict, sans texte avant/après, sans balises \
markdown, au format exact suivant :

{{
  "type": "flowchart" | "schema" | "photo" | "table" | "text_scan" | "logo_or_decoration" | "other",
  "is_flowchart": true ou false,
  "description": "description synthétique en 2-4 phrases (utile seulement si ce n'est PAS un flowchart)",
  "flowchart_nodes": [
    {{"id": "1", "shape": "start|end|process|decision|io", "text": "texte dans la forme"}}
  ],
  "flowchart_edges": [
    {{"from": "1", "to": "2", "label": "oui/non/vide"}}
  ]
}}

Règles :
- "is_flowchart" = true seulement si l'image montre un VRAI logigramme
  (formes reliées par des flèches représentant un enchaînement d'étapes ou
  de décisions : rectangles = étapes, losanges = décisions, ovales =
  début/fin). Un simple schéma technique, une photo de pièce, un tableau ou
  une capture d'écran ne sont PAS des flowcharts.
- Si "is_flowchart" est false, laisse "flowchart_nodes" et "flowchart_edges" \
vides ([]).
- Base-toi uniquement sur ce que tu vois réellement dans l'image. Le texte \
de contexte ci-dessous peut t'aider à comprendre le sujet mais ne doit \
JAMAIS influencer ta classification visuelle (ne déduis pas qu'il y a un \
logigramme juste parce que le texte parle d'une procédure).

Contexte textuel de la page (peut être vide) :
{page_text}
"""


def classify_and_extract_visual(image_bytes, page_text="", model_call=None):
    """
    Un seul appel au modèle de vision : renvoie le type de contenu, et si
    c'est un logigramme, sa structure (nœuds/arêtes) directement.

    model_call : callable(image_bytes: bytes, prompt: str) -> str
        A brancher sur le client déjà utilisé ailleurs dans le pipeline
        (celui qui alimente generate_caption_for_page /
        generate_caption_for_image). Voir _default_model_call() plus bas
        pour un exemple avec le SDK Anthropic.

    Retourne un dict :
        {
          "type": str,
          "is_flowchart": bool,
          "description": str,
          "flowchart_nodes": list,
          "flowchart_edges": list,
        }
    """
    if model_call is None:
        model_call = _default_model_call

    prompt = _VISION_PROMPT.format(page_text=(page_text[:800] if page_text else "(aucun)"))

    try:
        raw = model_call(image_bytes, prompt)
    except Exception as e:
        print(f"⚠️ Appel modèle de vision échoué ({e}) -> fallback CV assoupli")
        is_diag = _legacy_cv_fallback(image_bytes)
        return {
            "type": "flowchart" if is_diag else "other",
            "is_flowchart": is_diag,
            "description": "",
            "flowchart_nodes": [],
            "flowchart_edges": [],
        }

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("⚠️ Réponse du modèle non-JSON, on traite la page comme classique")
        return {
            "type": "other",
            "is_flowchart": False,
            "description": raw[:500],
            "flowchart_nodes": [],
            "flowchart_edges": [],
        }

    data.setdefault("type", "other")
    data.setdefault("is_flowchart", False)
    data.setdefault("description", "")
    data.setdefault("flowchart_nodes", [])
    data.setdefault("flowchart_edges", [])
    return data


def get_visual_analysis(image_bytes, page_text="", model_call=None):
    """
    Point d'entrée unique à appeler UNE FOIS par image/page dans ingest_pdf.
    Fait le pré-filtre CV, puis (si pertinent) l'appel au modèle de vision.
    Toujours le même résultat réutilisé pour classifier ET extraire.
    """
    if not _quick_visual_signal(image_bytes):
        return {
            "type": "other",
            "is_flowchart": False,
            "description": "",
            "flowchart_nodes": [],
            "flowchart_edges": [],
        }
    return classify_and_extract_visual(image_bytes, page_text, model_call=model_call)


def is_diagram(image_bytes, page_text="", model_call=None):
    """
    Conservée pour compatibilité avec le reste du code (signature proche de
    l'originale). Préférez get_visual_analysis() dans le nouveau code pour
    éviter de refaire l'appel deux fois (classification + extraction).
    """
    return get_visual_analysis(image_bytes, page_text, model_call=model_call)["is_flowchart"]


def _default_model_call(image_bytes, prompt):
    """
    Exemple d'implémentation avec le SDK Anthropic. Remplacez par le client
    déjà utilisé ailleurs dans votre pipeline (Azure OpenAI vision, etc.) si
    ce n'est pas celui-ci — c'est tout l'intérêt du paramètre model_call.
    """
    import base64
    import anthropic

    client = anthropic.Anthropic()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ==============================================================================
# 3) INGEST_PDF — mêmes fonctions externes qu'avant (fitz, pdfplumber,
#    process_flowchart, graph_to_content, generate_caption_for_page,
#    generate_caption_for_image, embed_query, upload_to_azure_search,
#    upload_image_to_blob, ocr_full_document, chunk_text, upscale_if_needed
#    supposées déjà définies ailleurs dans votre module).
# ==============================================================================

def ingest_pdf(path, category="general"):
    print("🚀 START INGEST PDF")

    doc_fitz = fitz.open(path)

    with pdfplumber.open(path) as pdf:

        needs_ocr = any(
            not page.extract_text() or page.extract_text().strip() == ""
            for page in pdf.pages
        )
        ocr_pages = ocr_full_document(path) if needs_ocr else []

        page_texts = []
        for i, p in enumerate(pdf.pages):
            page_text = p.extract_text()
            if not page_text or not page_text.strip():
                page_text = ocr_pages[i] if i < len(ocr_pages) else ""
            page_texts.append(page_text)

        document_title = page_texts[0][:500] if page_texts else ""

        for page_num, page in enumerate(pdf.pages):

            text = page_texts[page_num]
            used_ocr = (
                page_num < len(ocr_pages) and text == ocr_pages[page_num]
            )

            if not text.strip():
                continue

            previous_text = page_texts[page_num - 1] if page_num > 0 else ""
            next_text = (
                page_texts[page_num + 1]
                if page_num < len(page_texts) - 1
                else ""
            )

            image_docs = []
            page_images_urls = []

            page_fitz = doc_fitz[page_num]
            image_list = page_fitz.get_images(full=True)

            page_docs = []

            # ------------------------------------------------------------
            # 📄 PAGE COMPLETE
            # ------------------------------------------------------------
            try:
                mat = fitz.Matrix(2.5, 2.5)
                pix = page_fitz.get_pixmap(matrix=mat, alpha=False)
                page_png = pix.tobytes("png")

                # Un seul appel, réutilisé pour la décision ET l'extraction
                page_analysis = get_visual_analysis(page_png, page_text=text)
                page_is_diagram = page_analysis["is_flowchart"]

                page_doc = None

                if page_is_diagram:
                    print(f"🔀 Logigramme pleine page détecté page {page_num + 1}")

                    # process_flowchart / graph_to_content restent vos
                    # fonctions existantes (extraction fine du graphe).
                    mygraph = process_flowchart(page_png)
                    caption = graph_to_content(mygraph)

                    if caption and len(caption.split()) >= 5:
                        page_blob_url = upload_image_to_blob(page_png)
                        page_images_urls.append(page_blob_url)

                        sentence = f"""
                        Logigramme technique de diagnostic.

                        Source :
                        {os.path.basename(path)}

                        Page :
                        {page_num + 1}

                        Structure du logigramme :
                        {caption}

                        Contexte textuel de la page :
                        {text[:1500]}
                        """

                        page_doc = {
                            "id": str(uuid.uuid4()),
                            "content": caption,
                            "search_text": sentence,
                            "content_vector": embed_query(sentence),
                            "source_file": os.path.basename(path),
                            "page_number": page_num + 1,
                            "doc_type": "pdf_flowchart",
                            "category": category or "general",
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "image_urls": [page_blob_url],
                            "image_text": text[:2000],
                        }
                    else:
                        # 🩹 Filet de sécurité : détecté comme logigramme
                        # mais extraction du graphe inexploitable -> on ne
                        # perd pas la page, on retombe sur une description
                        # classique.
                        print(
                            f"⚠️ Logigramme détecté page {page_num + 1} mais "
                            f"graphe inexploitable -> fallback description classique"
                        )
                        page_is_diagram = False

                if not page_is_diagram:
                    page_blob_url = upload_image_to_blob(page_png)
                    page_description = generate_caption_for_page(
                        image_bytes=page_png,
                        page_text=text[:2000],
                        previous_page_text=previous_text[:1000],
                        next_page_text=next_text[:1000],
                    )

                    if page_description:
                        sentence = f"""
                        Document technique.

                        Contexte général du document :
                        {document_title}

                        Description de la page :
                        {page_description}

                        Texte de la page :
                        {text[:1500]}
                        """

                        page_doc = {
                            "id": str(uuid.uuid4()),
                            "content": page_description,
                            "search_text": sentence,
                            "content_vector": embed_query(sentence),
                            "source_file": os.path.basename(path),
                            "page_number": page_num + 1,
                            "doc_type": "pdf_page",
                            "category": category or "general",
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "image_urls": [page_blob_url],
                            "image_text": text[:2000],
                        }

                if page_doc:
                    page_docs.append(page_doc)

            except Exception as e:
                print(f"Erreur analyse page complète {page_num + 1}: {e}")

            # ------------------------------------------------------------
            # 🖼️ IMAGES EMBARQUEES DANS LA PAGE
            # ------------------------------------------------------------
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc_fitz.extract_image(xref)
                    image_bytes = base_image["image"]

                    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
                    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Impossible de décoder l'image extraite du PDF.")

                    image = upscale_if_needed(image)
                    success, encoded_image = cv2.imencode(".png", image)
                    if not success:
                        raise ValueError("Impossible d'encoder l'image en PNG.")
                    image_bytes = encoded_image.tobytes()

                    blob_url = upload_image_to_blob(image_bytes)

                    # Un seul appel, réutilisé pour la décision ET la caption
                    # (plus de double appel à is_diagram comme avant)
                    img_analysis = get_visual_analysis(image_bytes, page_text=text)
                    image_is_diagram = img_analysis["is_flowchart"]

                    if image_is_diagram:
                        mygraph = process_flowchart(image_bytes)
                        caption = graph_to_content(mygraph)
                        if not caption or len(caption.split()) < 5:
                            # même filet de sécurité que pour la page entière
                            print(
                                f"⚠️ Image {img_index} page {page_num + 1} détectée "
                                f"logigramme mais graphe inexploitable -> fallback caption"
                            )
                            image_is_diagram = False
                            caption = generate_caption_for_image(image_bytes)
                    else:
                        caption = generate_caption_for_image(image_bytes)

                    if not caption:
                        continue
                    if "logo" in caption.lower():
                        continue
                    if len(caption.split()) < 5:
                        continue

                    h, w = image.shape[:2]
                    if (w < 100 or h < 100) and len(caption.split()) < 10:
                        continue

                    page_images_urls.append(blob_url)

                    sentence = f"Schéma technique. {caption}"
                    image_doc = {
                        "id": str(uuid.uuid4()),
                        "content": caption,
                        "search_text": sentence,
                        "content_vector": embed_query(sentence),
                        "source_file": os.path.basename(path),
                        "page_number": page_num + 1,
                        "doc_type": "pdf_flowchart" if image_is_diagram else "pdf_image",
                        "category": category or "general",
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "image_url": blob_url,
                        "image_text": text[:500],
                    }

                    # Avant : ajouté seulement si is_diagram (bug -> les
                    # images NON-logigramme n'étaient jamais indexées ici).
                    # Correction : on indexe l'image dans tous les cas où on
                    # a un caption exploitable, en gardant le doc_type pour
                    # distinguer logigramme vs image classique en recherche.
                    image_docs.append(image_doc)

                except Exception as e:
                    print(f"Erreur image page {page_num}: {e}")
                    continue

            # ------------------------------------------------------------
            # ✂️ CHUNKING TEXTE (procédures textuelles, avec ou sans image)
            # ------------------------------------------------------------
            chunks = chunk_text(text)
            text_docs = []

            # Détection légère "texte procédural" (étapes numérotées) pour
            # enrichir le doc_type -> utile pour prioriser ces chunks en
            # recherche même quand il n'y a ni image ni logigramme.
            is_procedural_text = bool(
                re.search(r"(?im)^\s*(étape\s*\d+|[0-9]+[\.\)]\s+\S)", text)
            )

            for chunk in chunks:
                sentence = f"""
                Document technique.
                Source: {path.split("/")[-1]}
                Contenu: {chunk}
                """

                text_docs.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": chunk,
                        "search_text": sentence,
                        "content_vector": embed_query(sentence),
                        "source_file": os.path.basename(path),
                        "page_number": page_num + 1,
                        "doc_type": "pdf_procedure_text" if is_procedural_text else "pdf_text",
                        "category": category or "general",
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "image_urls": page_images_urls,
                        "used_ocr": used_ocr,
                        "has_images": len(page_images_urls) > 0,
                    }
                )

            # ------------------------------------------------------------
            # 🚀 UPLOAD
            # ------------------------------------------------------------
            if text_docs:
                upload_to_azure_search(text_docs)
            if page_docs:
                upload_to_azure_search(page_docs)
            if image_docs:
                upload_to_azure_search(image_docs)
