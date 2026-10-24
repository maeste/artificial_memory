# Checklist manuale Fedora — verifica O5

Copiata da untracked_doc/08_EVALUATION.md §6. Per ogni voce aggiungi data,
ambiente, PASS/FAIL/non eseguito ed evidenza breve. Le voci non eseguite non
si spuntano. Scope aggiornato 2026-10-24: il gate S2-audio è PASS con voce reale
(run-3, decisione utente opzione A); le voci microfono della checklist sono
a tutti gli effetti parte del percorso verificato.

Ambiente: Fedora 44, GNOME 50.5, Wayland, RTX 5000 Ada. Data: da compilare.

Copia questa checklist in `docs/evaluation/manual.md`; per ogni voce aggiungi data, ambiente, PASS/FAIL/non eseguito ed evidenza breve. Le voci non eseguite non si spuntano.

- [ ] Avvio reale, stato backend/device/revisione visibili, errori preflight comprensibili.
- [ ] Directory personale di prova con i quattro formati indicizzata; seconda scansione invariata.
- [ ] Messaggio nuovo da agente realmente collegato produce ricordo pertinente.
- [ ] Frase nel microfono produce un ricordo corretto; pausa mic lascia attivo il testo.
- [ ] Cambio argomento e input audio/testo diversi mantengono suggerimenti interpretabili; registrare anche risultati indesiderati.
- [ ] Query negativa lascia HUD vuoto; show-all DEBUG distingue la diagnostica dall'uso normale.
- [ ] Pin A e B, tray visibile, rimozione, associazione verificata nella ricerca successiva.
- [ ] Apri MP4 al secondo atteso, PDF alla pagina attesa, immagine e Markdown col programma di sistema.
- [ ] HUD aperto tardi recupera lo stato; reset non ripristina pin vecchi.
- [ ] Export off: audio/allegati presenti, nessun download ASR, percorso e clipboard riportati correttamente.
- [ ] Export local: modello locale predisposto, parlato selezionato trascritto, finestra reattiva durante il lavoro.
- [ ] Cambio opzione trascrizione vale per export successivo e compare nel manifest.
- [ ] Sorgente modificata dopo pin segnalata; export resta utile con testo congelato e avvisi.
- [ ] Dopo arresto, pacchetto esportato ancora leggibile senza runtime.
- [ ] Passaggio manuale a un agente esterno: con off fornire anche gli allegati; con local leggere le trascrizioni. Registrare quali modalità l'agente destinatario può effettivamente leggere, senza promettere integrazione automatica.

