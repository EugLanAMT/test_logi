
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
 
