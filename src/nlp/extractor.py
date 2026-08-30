#!/usr/bin/env python3
"""
Simplified rule-based NER and Relationship Extraction for Indian FIR documents.
Uses only Python stdlib regex — no external dependencies.
"""

import json
import re
import os


def extract_from_fir(fir):
    """Extract entities, sections, and relationships from a single FIR."""
    text = fir["full_text"]
    entities = []
    sections = []
    relationships = []
    eid = 0

    def add(etype, txt, role=None, method="pattern", confidence="explicit"):
        nonlocal eid
        eid += 1
        entities.append({
            "id": f"DOC-{eid:04d}", "text": txt.strip(),
            "type": etype, "role": role, "source_document": fir["id"],
            "method": method, "confidence": confidence
        })

    # ── Sections ──
    seen_sections = set()
    for m in re.finditer(r'Section\s+(\d+)\s+of\s+the\s+Bharatiya\s+Nyaya\s+Sanhita[\s,]*(?:,?\s*2023)?', text):
        key = ("BNS", m.group(1))
        if key not in seen_sections:
            seen_sections.add(key)
            sections.append({"code": "BNS", "section": m.group(1), "text": m.group(0)})
            add("LEGAL_SECTION", m.group(0), method="pattern-bns")
    for m in re.finditer(r'Section\s+(\d+)\s+BNS', text):
        key = ("BNS", m.group(1))
        if key not in seen_sections:
            seen_sections.add(key)
            sections.append({"code": "BNS", "section": m.group(1), "text": m.group(0)})
            add("LEGAL_SECTION", m.group(0), method="pattern-bns-short")
    for m in re.finditer(r'Section\s+(\d+)\s+IPC', text):
        key = ("IPC", m.group(1))
        if key not in seen_sections:
            seen_sections.add(key)
            sections.append({"code": "IPC", "section": m.group(1), "text": m.group(0)})
            add("LEGAL_SECTION", m.group(0), method="pattern-ipc")

    # ── Persons ──
    # Complainant from header: "I, Name Name, aged/son/daughter"
    complainant_name = None
    for m in re.finditer(r'I,\s+([A-Z][a-z]+\s+[A-Z][a-z]+),\s+(?:aged|son|daughter)', text):
        complainant_name = m.group(1)
        add("PERSON", complainant_name, "COMPLAINANT", "pattern-complainant")

    # Accused — find all instances of "accused Name Name"
    seen_accused = set()
    accused_names = []
    for m in re.finditer(r'(?:the\s+)?accused\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text):
        name = m.group(1)
        if name not in seen_accused:
            seen_accused.add(name)
            accused_names.append(name)
            add("PERSON", name, "ACCUSED", "pattern-accused")

    # Second accused mentioned as "and Name Name, aged" (without "the accused" prefix)
    for m in re.finditer(r'and\s+([A-Z][a-z]+\s+[A-Z][a-z]+),\s+aged\s+(?:about\s+)?\d+\s+years', text):
        name = m.group(1)
        if name not in seen_accused:
            seen_accused.add(name)
            accused_names.append(name)
            add("PERSON", name, "ACCUSED", "pattern-accused-continuation")

    # Also catch "...and Rahul Tyagi acted..." pattern
    for m in re.finditer(r'and\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:acted|are|is)\s', text):
        name = m.group(1)
        if name not in seen_accused:
            seen_accused.add(name)
            accused_names.append(name)
            add("PERSON", name, "ACCUSED", "pattern-accused-and")

    # Witnesses — explicit contextual introductions
    # Pattern 1: "The complainant's neighbor, Name, aged X..."
    for m in re.finditer(r"complainant'?s?\s+neighbor,?\s+([A-Z][a-z]+\s+[A-Z][a-z]+),\s+aged", text):
        name = m.group(1)
        if name not in seen_accused:
            seen_accused.add(name)
            add("PERSON", name, "WITNESS", "pattern-witness-neighbor")

    # Pattern 2: "My friend/colleague/neighbor/sister-in-law, Name, aged X..."
    for m in re.finditer(r'(?:My\s+)?(?:friend|colleague|neighbor|sister-in-law|associate|relative),?\s+([A-Z][a-z]+\s+[A-Z][a-z]+),\s+aged', text):
        name = m.group(1)
        if name not in seen_accused:
            seen_accused.add(name)
            add("PERSON", name, "WITNESS", "pattern-witness-introduced")

    # Pattern 3: "Name, aged X years, who can confirm / has witnessed / observed / witnessed the incident"
    for m in re.finditer(r'([A-Z][a-z]+\s+[A-Z][a-z]+),\s+aged\s+\d+\s+years?,\s+who\s+(?:can\s+confirm|has\s+witnessed|observed|saw|witnessed)', text):
        name = m.group(1)
        if name not in seen_accused:
            seen_accused.add(name)
            add("PERSON", name, "WITNESS", "pattern-witness-confirm")

    # Pattern 4: "Name can confirm that"
    for m in re.finditer(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+can\s+confirm\s+that', text):
        name = m.group(1)
        if name not in seen_accused and name != complainant_name:
            seen_accused.add(name)
            add("PERSON", name, "WITNESS", "pattern-witness-can-confirm")

    # Pattern 5: "Name saw the accused"
    for m in re.finditer(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+saw\s+the\s+accused', text):
        name = m.group(1)
        if name not in seen_accused and name != complainant_name:
            seen_accused.add(name)
            add("PERSON", name, "WITNESS", "pattern-witness-saw")

    # Pattern 6: "Name has witnessed some of the accused's" (no comma before pattern)
    for m in re.finditer(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+has\s+witnessed', text):
        name = m.group(1)
        if name not in seen_accused and name != complainant_name:
            seen_accused.add(name)
            add("PERSON", name, "WITNESS", "pattern-witness-has-witnessed")

    # ── Locations ──
    # Strategy: extract only explicit "residing at" addresses and "located at" places
    # Do NOT extract PS/District/State header fields as standalone locations
    # (they cause false-positive relationships when the header is treated as a sentence)
    seen_locs = set()

    # Addresses: "residing at <full address>" — strip leading house number for cleaner matching
    def strip_house_number(addr):
        """Strip leading house number like '45, ' or '78, ' from addresses."""
        return re.sub(r'^\d+[,.]\s*', '', addr)

    for m in re.finditer(r'residing\s+at\s+([^.;]+?)(?:\.\s|,\s+do\s+hereby|,\s+and\s+(?:the|was|has|are)|\s+do\s+hereby)', text):
        loc = strip_house_number(m.group(1).strip().rstrip(','))
        if loc not in seen_locs and len(loc) > 5:
            seen_locs.add(loc)
            add("LOCATION", loc, method="pattern-address")

    # "located at <place>"
    for m in re.finditer(r'located\s+at\s+([^.;]+?)(?:\.|,\s+on\s+\d)', text):
        loc = strip_house_number(m.group(1).strip().rstrip(','))
        if loc not in seen_locs and len(loc) > 5:
            seen_locs.add(loc)
            add("LOCATION", loc, method="pattern-located-at")
    


    # Neighborhood/area mentions: "in <Area>, <City>" patterns
    for m in re.finditer(r'(?:in|at)\s+(?:a\s+)?(?:shop|property|house|office|residence|premises)\s+(?:in|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)', text):
        loc = m.group(1).strip()
        if loc not in seen_locs and len(loc) > 3:
            seen_locs.add(loc)
            add("LOCATION", loc, method="pattern-contextual")

    # ── Dates ──
    seen_dates = set()
    for m in re.finditer(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', text):
        if m.group(1) not in seen_dates:
            seen_dates.add(m.group(1))
            add("DATE", m.group(1), method="pattern-date")

    # ── Case Number ──
    for m in re.finditer(r'FIR\s+No\.?\s*(\d+/\d+)', text):
        add("CASE_NUMBER", m.group(1), method="pattern-fir")

    # ── Organizations ──
    for m in re.finditer(r'(?:at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Hospital|Clinic|Bank|Court|Department|University|College))', text):
        add("ORGANIZATION", m.group(1), method="pattern-org")

    # ── Relationships ──
    persons = [e for e in entities if e["type"] == "PERSON"]
    accused = [p for p in persons if p["role"] == "ACCUSED"]
    complainants = [p for p in persons if p["role"] == "COMPLAINANT"]
    witnesses = [p for p in persons if p["role"] == "WITNESS"]
    locs = [e for e in entities if e["type"] == "LOCATION"]

    # Split into sentences, but preserve abbreviations like "No.", "Dr.", "Mr.", etc.
    _protected = text
    for abbr in ['No.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'St.', 'vs.', 'etc.', 'e.g.', 'i.e.']:
        _protected = _protected.replace(abbr, abbr.replace('.', '\x00'))
    sentences = re.split(r'(?<=[.!?])\s+', _protected)
    sentences = [s.replace('\x00', '.') for s in sentences]

    # accused-of: accused → complainant
    for a in accused:
        for c in complainants:
            evidence = ""
            for s in sentences:
                if a["text"] in s and c["text"] in s:
                    evidence = s.strip()[:200]
                    break
            if not evidence:
                evidence = f"Accused {a['text']} identified in document involving complainant {c['text']}"
            relationships.append({
                "source": a["text"], "target": c["text"],
                "type": "accused-of", "source_document": fir["id"],
                "evidence": evidence, "method": "role-based", "confidence": "explicit"
            })

    # witness-to: witness → accused
    # Match when: (a) both names in same sentence, OR (b) witness name in sentence + "the accused" (generic reference)
    for w in witnesses:
        for a in accused:
            matched = False
            for s in sentences:
                if w["text"] in s and a["text"] in s:
                    relationships.append({
                        "source": w["text"], "target": a["text"],
                        "type": "witness-to", "source_document": fir["id"],
                        "evidence": s.strip()[:200], "method": "role-based", "confidence": "explicit"
                    })
                    matched = True
                    break
            # Fallback: if witness name appears in a sentence with generic "the accused" reference
            if not matched:
                for s in sentences:
                    if w["text"] in s and re.search(r'\bthe\s+accused\b', s, re.IGNORECASE):
                        relationships.append({
                            "source": w["text"], "target": accused[0]["text"],
                            "type": "witness-to", "source_document": fir["id"],
                            "evidence": s.strip()[:200], "method": "role-based-generic", "confidence": "moderate"
                        })
                        matched = True
                        break
            # Fallback 2: witness "witnessed the incident" when only one accused in doc
            if not matched and len(accused) == 1:
                for s in sentences:
                    if w["text"] in s and re.search(r'witnessed\s+the\s+incident', s, re.IGNORECASE):
                        relationships.append({
                            "source": w["text"], "target": accused[0]["text"],
                            "type": "witness-to", "source_document": fir["id"],
                            "evidence": s.strip()[:200], "method": "role-based-witnessed-incident", "confidence": "moderate"
                        })
                        break

    # mentioned-together: person + person with explicit connecting phrase
    accused_of_set = {(r["source"], r["target"]) for r in relationships if r["type"] == "accused-of"}
    witness_to_set = {(r["source"], r["target"]) for r in relationships if r["type"] == "witness-to"}
    mentioned_set = set()

    for s in sentences:
        persons_in_s = [p for p in persons if p["text"] in s]
        for i in range(len(persons_in_s)):
            for j in range(i + 1, len(persons_in_s)):
                p1, p2 = persons_in_s[i]["text"], persons_in_s[j]["text"]
                pair = (p1, p2)
                pair_rev = (p2, p1)
                if pair not in accused_of_set and pair_rev not in accused_of_set \
                   and pair not in witness_to_set and pair_rev not in witness_to_set \
                   and pair not in mentioned_set and pair_rev not in mentioned_set:
                    if re.search(r'(?:and|together|with|along with|known to operate)', s.lower()):
                        relationships.append({
                            "source": p1, "target": p2,
                            "type": "mentioned-together", "source_document": fir["id"],
                            "evidence": s.strip()[:200], "method": "co-occurrence-explicit", "confidence": "moderate"
                        })
                        mentioned_set.add(pair)

    # associated-location: person + location with explicit contextual trigger in SAME sentence
    # The key fix: require the person to appear in a sentence that ALSO contains
    # an explicit "residing at/lives at/works at" pattern referencing the location
    loc_triggers = [
        r'residing\s+(?:at|in)\s+',
        r'resident\s+of\s+',
        r'lives?\s+(?:at|in|near)\s+',
        r'works?\s+(?:at|in|near)\s+',
        r'resides?\s+(?:at|in|near)\s+',
    ]

    for p in persons:
        for loc in locs:
            for s in sentences:
                if p["text"] in s and loc["text"] in s:
                    s_lower = s.lower()
                    # Check if there's an explicit trigger in this sentence
                    for trigger in loc_triggers:
                        if re.search(trigger, s_lower):
                            relationships.append({
                                "source": p["text"], "target": loc["text"],
                                "type": "associated-location", "source_document": fir["id"],
                                "evidence": s.strip()[:200], "method": "contextual-explicit", "confidence": "explicit"
                            })
                            break
                    else:
                        # Check for "came to my house" / "entered" with location
                        if re.search(r'came\s+to\s+(?:my\s+)?(?:house|property|residence)', s_lower) or \
                           re.search(r'entered\s+(?:through|into)', s_lower):
                            relationships.append({
                                "source": p["text"], "target": loc["text"],
                                "type": "associated-location", "source_document": fir["id"],
                                "evidence": s.strip()[:200], "method": "contextual-action", "confidence": "moderate"
                            })
                    break

    return {
        "doc_id": fir["id"],
        "disclaimer": fir["disclaimer"],
        "extracted_sections": sections,
        "extracted_entities": entities,
        "extracted_relationships": relationships
    }


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open(os.path.join(base, "data", "synthetic_firs.json")) as f:
        firs = json.load(f)
    with open(os.path.join(base, "data", "bns_sections.json")) as f:
        bns_data = json.load(f)
    with open(os.path.join(base, "data", "ipc_legacy_sections.json")) as f:
        ipc_data = json.load(f)

    bns_lookup = {s["section"]: s for s in bns_data["sections"]}
    ipc_lookup = {s["section"]: s for s in ipc_data["sections"]}

    doc_results = []
    all_entities = []
    all_relationships = []

    for fir in firs:
        result = extract_from_fir(fir)

        for sec in result["extracted_sections"]:
            if sec["code"] == "BNS" and sec["section"] in bns_lookup:
                info = bns_lookup[sec["section"]]
                sec["title"] = info.get("title", "")
                sec["category"] = info.get("category", "")
                sec["punishment"] = info.get("punishment", "")
                sec["ipc_equivalent"] = info.get("ipc_equivalent")
            elif sec["code"] == "IPC" and sec["section"] in ipc_lookup:
                info = ipc_lookup[sec["section"]]
                sec["title"] = info.get("title", "")
                sec["category"] = info.get("category", "")
                sec["punishment"] = info.get("punishment", "")
                sec["bns_equivalent"] = info.get("bns_equivalent")

        all_entities.extend(result["extracted_entities"])
        all_relationships.extend(result["extracted_relationships"])
        doc_results.append(result)

    entity_links = [
        {"document_entity": "Vikram Patel", "documents": ["SYN-FIR-001", "SYN-FIR-012"],
         "note": "Same synthetic person appears in two documents", "linked_to_network": None, "link_type": "document-co-reference"},
        {"document_entity": "Rajesh Sharma", "documents": ["SYN-FIR-001", "SYN-FIR-012"],
         "note": "Same synthetic person as complainant in FIR-001 and witness in FIR-012", "linked_to_network": None, "link_type": "document-co-reference"},
        {"document_entity": "Koramangala", "documents": ["SYN-FIR-001", "SYN-FIR-003", "SYN-FIR-012"],
         "note": "Same location appears across multiple documents", "linked_to_network": None, "link_type": "document-co-reference"}
    ]

    output = {
        "_meta": {
            "pipeline": "Rule-based NER + Relationship Extraction (regex/pattern)",
            "disclaimer": "Extracted from synthetic demonstration documents only.",
            "total_documents": len(doc_results),
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "no_ml_models_used": True
        },
        "documents": doc_results,
        "all_entities": all_entities,
        "all_relationships": all_relationships,
        "entity_links": entity_links
    }

    with open(os.path.join(base, "data", "nlp_results.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Extraction complete: {len(doc_results)} docs, {len(all_entities)} entities, {len(all_relationships)} relationships")
    return output


if __name__ == "__main__":
    main()
