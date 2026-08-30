"""Grounded evaluation set for the RTI RAG system.

15 question / ground-truth pairs, all derived from the actual indexed corpus:
  - RTI Act 2005            (processed/rti_act_2005_final.txt)
  - RTI Act 2005 (consolidated 01.02.2011)
  - Delhi RTI Act 2001      (processed/delhi_rti_2001_ocr_final.txt)
  - Delhi HC Judgment 22.01.2021  (processed/hc_judgment_2021_final.txt)
  - CIC Decision 23.04.2026       (processed/cic_decision_2026_final.txt)

Categories:
  factual        - 5  simple section-content questions
  graph          - 5  questions whose answer lives in the case documents
                   (exercises the citation-graph expansion path)
  multi_hop      - 3  questions requiring comparing two documents
  out_of_corpus  - 2  genuinely unanswerable from the corpus (must trigger the
                   "I don't have enough information" graceful stop, not a
                   hallucination)
"""

EVAL_SET = [
    # ---- 5 simple factual questions -------------------------------------
    {
        "id": "F1",
        "category": "factual",
        "question": "How long does a CPIO or SPIO have to respond to an RTI application, "
                    "and what is the special deadline when the information concerns a "
                    "person's life or liberty?",
        "ground_truth": "Section 7(1) of the RTI Act 2005 requires the CPIO/SPIO to provide "
                        "the information, or reject the request, within 30 days of receiving "
                        "the request. Where the information concerns the life or liberty of a "
                        "person, it must be provided within 48 hours of the receipt of the "
                        "request.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    {
        "id": "F2",
        "category": "factual",
        "question": "What are the two stages of appeal under the RTI Act 2005 and what are "
                    "their respective time limits?",
        "ground_truth": "Under section 19 of the RTI Act 2005, a first appeal is made within "
                        "30 days to the officer senior in rank to the CPIO/SPIO (section 19(1)). "
                        "A second appeal then lies within 90 days to the Central or State "
                        "Information Commission (section 19(3)).",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    {
        "id": "F3",
        "category": "factual",
        "question": "What penalty can the Information Commission impose on a CPIO who refuses "
                    "an application without reasonable cause, and what is the maximum amount?",
        "ground_truth": "Section 20(1) of the RTI Act 2005 allows the Commission to impose a "
                        "penalty of 250 rupees for each day until the application is received "
                        "or the information is furnished, but the total penalty must not exceed "
                        "25,000 rupees.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    {
        "id": "F4",
        "category": "factual",
        "question": "Under section 8(1) of the RTI Act 2005, what is the exemption for "
                    "information that would harm the competitive position of a third party?",
        "ground_truth": "Section 8(1)(d) exempts information including commercial confidence, "
                        "trade secrets, or intellectual property, where disclosure would harm "
                        "the competitive position of a third party, unless the competent "
                        "authority is satisfied that the larger public interest warrants the "
                        "disclosure.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    {
        "id": "F5",
        "category": "factual",
        "question": "If an RTI application is made to a public authority but the information is "
                    "held by, or more closely connected to, another public authority, how soon "
                    "must the first authority transfer the application?",
        "ground_truth": "Under section 6(3) of the RTI Act 2005, the public authority to which "
                        "the application is made must transfer the application (or the relevant "
                        "part) to the other authority and inform the applicant immediately; the "
                        "transfer must be made as soon as practicable and in no case later than "
                        "five days from the date of receipt of the application.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    # ---- 5 graph-expansion-triggering questions -------------------------
    {
        "id": "G1",
        "category": "graph",
        "question": "In the Union Bank of India RTI case, on what ground did the CPIO "
                    "initially refuse the applicant's request for the board approval note?",
        "ground_truth": "In the Delhi HC judgment of 22 January 2021 (Union Bank of India case), "
                        "the CPIO refused the copy of the board note on the ground that it was "
                        "an internal document of the Bank and of commercial confidence, and was "
                        "therefore exempt from disclosure under section 8(1)(d) of the RTI Act.",
        "expected_graph_expansion": True,
        "expected_graceful_stop": False,
    },
    {
        "id": "G2",
        "category": "graph",
        "question": "What penalty did the CIC originally impose on each CPIO in the Union Bank "
                    "case, and what did the High Court ultimately reduce it to?",
        "ground_truth": "The CIC, by its final order dated 14 December 2020, imposed a penalty "
                        "of Rs 10,000 each on the then CPIO and the present CPIO. The High "
                        "Court of Delhi, by its judgment dated 22 January 2021, upheld the "
                        "penalty but reduced it to Rs 5,000 each.",
        "expected_graph_expansion": True,
        "expected_graceful_stop": False,
    },
    {
        "id": "G3",
        "category": "graph",
        "question": "Why did the CIC find mala fides on the part of the petitioners in the "
                    "Union Bank of India case?",
        "ground_truth": "The CIC found mala fides because the CPIOs changed their stand: "
                        "initially they said the board note could not be disclosed as it was "
                        "exempt under section 8(1)(d) of the RTI Act, but after the show-cause "
                        "notice they claimed the document was not traceable on record, an "
                        "inconsistent and unreasonable change in position.",
        "expected_graph_expansion": True,
        "expected_graceful_stop": False,
    },
    {
        "id": "G4",
        "category": "graph",
        "question": "In the CIC decision on Prasanta Kumar Sahoo, what information was sought "
                    "and on what ground did the CPIO refuse it?",
        "ground_truth": "In the CIC decision dated 23 April 2026 (Prasanta Kumar Sahoo v. CPIO, "
                        "Ministry of Labour and Employment), the appellant sought a copy of the "
                        "CVC investigation report (CVC complaint No. 19309/2023). The CPIO "
                        "refused it as personal information under section 8(1)(j) of the RTI Act "
                        "2005, relying on DOPT's office memorandum dated 14 August 2013.",
        "expected_graph_expansion": True,
        "expected_graceful_stop": False,
    },
    {
        "id": "G5",
        "category": "graph",
        "question": "What did the CIC direct the CPIO to do in the Prasanta Kumar Sahoo case, "
                    "even though it declined to release the full investigation report?",
        "ground_truth": "The CIC held the refusal to share the full report legally sound but, "
                        "in the interest of administrative transparency, directed the CPIO to "
                        "provide a revised reply within 15 days, strictly limited to disclosing "
                        "the broad outcome of the investigation without the report itself or its "
                        "underlying records, and to send a compliance report to the Commission "
                        "within 7 days thereafter.",
        "expected_graph_expansion": True,
        "expected_graceful_stop": False,
    },
    # ---- 3 multi-hop / comparison questions -----------------------------
    {
        "id": "M1",
        "category": "multi_hop",
        "question": "Compare the time limit within which information must be furnished under "
                    "the Delhi RTI Act 2001 with the time limit under the Central RTI Act 2005.",
        "ground_truth": "The Delhi RTI Act 2001 (section 5) requires information to be furnished "
                        "as soon as practicable but normally within 15 days, and in any case "
                        "within 30 days of the receipt of the application. The Central RTI Act "
                        "2005 (section 7(1)) provides a single limit of 30 days (48 hours where "
                        "life or liberty is involved). So the Delhi Act's normal target is 15 "
                        "days, whereas the Central Act's general target is 30 days.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    {
        "id": "M2",
        "category": "multi_hop",
        "question": "How does the penalty for a CPIO differ between the Central RTI Act 2005 "
                    "and the Delhi RTI Act 2001?",
        "ground_truth": "Under section 20(1) of the Central RTI Act 2005, the Commission may "
                        "impose a monetary penalty of 250 rupees per day (capped at 25,000 "
                        "rupees) on a CPIO who refuses or delays without reasonable cause. The "
                        "Delhi RTI Act 2001 does not prescribe a fixed monetary fine; instead "
                        "the person responsible for furnishing information is personally liable, "
                        "and the remedy is disciplinary action under the applicable service "
                        "rules.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    {
        "id": "M3",
        "category": "multi_hop",
        "question": "How do the exemption grounds for commercial and trade secrets differ "
                    "between the Central RTI Act 2005 and the Delhi RTI Act 2001?",
        "ground_truth": "The Central RTI Act 2005 (section 8(1)(d)) exempts commercial "
                        "confidence, trade secrets, and intellectual property, and allows "
                        "disclosure if the competent authority is satisfied that the larger "
                        "public interest warrants it. The Delhi RTI Act 2001 (section 6, clause "
                        "on trade and commercial secrets) exempts 'trade and commercial secrets "
                        "or any other information protected by law', but does not include an "
                        "express larger-public-interest override.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": False,
    },
    # ---- 2 genuinely out-of-corpus questions ----------------------------
    {
        "id": "O1",
        "category": "out_of_corpus",
        "question": "What does the RTI Act 2005 say about the regulation of cryptocurrency, "
                    "Bitcoin, or digital currency by the Reserve Bank of India?",
        "ground_truth": "The RTI Act 2005 contains no provision on the regulation of "
                        "cryptocurrency, Bitcoin, or digital currency by the Reserve Bank of "
                        "India. The knowledge base does not cover this topic, so the system "
                        "should state that it does not have enough information rather than "
                        "fabricate an answer.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": True,
    },
    {
        "id": "O2",
        "category": "out_of_corpus",
        "question": "What punishment does the RTI Act 2005 prescribe for the crime of "
                    "kidnapping a person for ransom?",
        "ground_truth": "The RTI Act 2005 does not prescribe any punishment for the criminal "
                        "offence of kidnapping for ransom; that is criminal law, not covered by "
                        "the RTI Act. The knowledge base has no such information, so the system "
                        "should say it does not have enough information rather than invent a "
                        "penalty.",
        "expected_graph_expansion": False,
        "expected_graceful_stop": True,
    },
]
