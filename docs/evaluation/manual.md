# Report finale — sessione manuale GNOME 2026-10-24

Sessione live completa con l'utente al desktop (Fedora 44, GNOME 50.5, Wayland,
RTX 5000 Ada, whisper-base int8 CPU). App avviata con `run_poc.py --backend
real --runtime-root ./runtime`, DB `data/brain-real.db` con il corpus 08.

## Percorso utente verificato end-to-end (tutto PASS reale)

1. **Avvio**: preflight verde, header `backend real`, calibrazione congelata
   caricata (dopo fix — vedi bug #5), watcher 16/16 unchanged a ogni riavvio
   (zero re-embedding, ripetuto 5+ volte).
2. **Ricordi dal testo**: frase nel bridge → card `f01.md` rank 1 (cos 0.844);
   canary → `f05.pdf` (0.786, pagina 2).
3. **Ricordi dal microfono**: C1–C4 dette a voce reale → **gate Fase 1
   riaperto e superato 4/4** (run-3, `probe-real-voice-run3.json`); card live
   da voce; mic off blocca solo l'audio, il testo resta attivo.
4. **Selezione associativa**: pin doppio f01.md+f05.pdf → **arco simmetrico
   0.25 nel DB**, embedding invariati (norma 1.0) — S6 dal vivo; "boost grafo"
   visibile nelle card; "Pinnato" disabilitato; Rimuovi ok.
5. **Aperture al punto esatto**: f01.md (xdg-open), f05.pdf (evince pag. 2),
   f13.mp4 (**mpv --start=40**, transizione blu→verde) — tutte coi click.
6. **Export OFF**: pacchetto con pin (testo congelato + contesto di
   selezione), 4 WAV utente in assets, clipboard copied, zero ASR.
7. **Export LOCAL**: trascrizione **fedele** del WAV utente ("quanto durano i
   risultati nella cache e quando li invalidiamo.") — whisper-base su voce
   naturale (il TTS era il limite, non il modello); degrado senza dir modello
   verificato prima (avviso esplicito + allegati conservati).
8. **Durevolezza**: pacchetti di sessioni precedenti leggibili senza runtime.

## Bug trovati e chiusi (5)

| # | Bug | Causa radice | Fix |
|---|---|---|---|
| 1 | Bottoni Pin/Apri inerti + flash | `_on_sub_tick` ricostruiva le card ogni 100 ms (bottoni distrutti fra press e release) | Agent-5: `HudLogic` separata dai widget, rebuild solo su cambio reale (100→99 test) |
| 2 | Tray visivamente instabile | stessa causa | stesso fix |
| 3 | Etichetta bottone Mic fissa | label non derivata da `sensor_status` | Agent-5: `mic_label()` in HudLogic (94 test) |
| 4 | Toast export mai visibile (traceback) | `add_toast` su ApplicationWindow | Agent-5: `Adw.ToastOverlay` + `publish_toast` + `export_outcome` pura (100 test) |
| 5 | **Soglie default al posto della calibrazione** | `run_poc.py` non autocaricava `data/calibration.json`: le card video giuste (cos 0.725) scartate da min_final 1.0 zscore | Orchestrator: autocarico calibrazione (--calibration come override) |

Più fix tail (NUL/banner PTY, Orchestrator su modulo Agent-4 offline) e
CLI `new-session` (Orchestrator). Tutti committati con test.

## Voci di checklist NON eseguite (dichiarate)

- Registrazione di una **sessione agente reale** di lungo corso con tool
  esterno (le prove sono state con log/echo e mic; il bridge `script` con
  shell interattiva si è rivelato rumoroso: da usare solo per catturare un
  tool vero).
- `show-all` DEBUG (non abilitato durante la sessione).
- Itinerario di passaggio a un **agente LLM esterno** del pacchetto esportato
  (il pacchetto è completo e leggibile; il passaggio vero dipende dal tool
  dell'utente).

## Conclusione

Il workflow completo del PRD — parlare/lavorare → ricordi emergono →
apertura al punto → pin → contesto esportato per un agente — **funziona dal
vivo** sulla macchina bersaglio, con pesi reali, soglie calibrate e trascrizione
locale. S1–S8 tutti con prova reale (S2-audio superato con voce umana, run-3).
I 5 bug emersi dalla sessione live sono chiusi con test.
