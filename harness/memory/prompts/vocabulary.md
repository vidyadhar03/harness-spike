## Task: search vocabulary for a location
You are given the memory context for one location in a film production. The director wants real-world visual references for it. Image archives are indexed by architectural, geographic, and cultural terms, not by screenplay language, so your job is to translate.

- script_phrases: the production's own words for this place, quoted from the context as written, including Hindi or Hinglish. 3 to 8 short phrases. Do not translate or paraphrase them.
- description: 2 to 4 sentences describing what the place looks like, drawn only from the context. This is what a researcher would read before searching. Do not invent details the context does not support.
- terms: 6 to 12 search terms an archive would index.
  - Prefer named building techniques, materials, landforms, vegetation, settlement types, crafts, periods, and regions over generic descriptions. "Stone and wood house with a slate roof in the hills" is not a search term; the named regional technique for that construction is.
  - Include a few broader terms that would work even if the specific one is wrong, and a few terms for the landscape and vegetation as well as the buildings.
  - Only propose a term if you are reasonably confident it is a real, documented name. Each term is checked against an encyclopedia and dropped if it does not exist, so a wrong guess costs nothing but an invented term you insist on wastes a slot.
  - kind: technique, material, landform, vegetation, settlement, craft, period, region, or other.
  - needs_region: true when the term is generic worldwide — a landform, material, or technique found in many countries — so an archive search for it alone returns the world's most photographed example rather than this region's. "river gorge", "slate roofing", "terraced orchard", and "dry stone masonry" need a region. False when the term is already specific: "Kath-kuni", "Kinnaur", "Chamba", and "Himachali vernacular" do not. Terms that need a region are searched inside the terms you mark kind region, so use kind region only for a named geographic area.
  - why: one short line on what this term is expected to show, and which part of the context it comes from.
- Do not name specific villages, towns, or landmarks as terms unless the context names them. Terms describe a type of place, not a particular address.
- Write terms in English unless the standard name is a transliteration, which is common for regional techniques and crafts.
