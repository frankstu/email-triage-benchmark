// Exportiert die neuesten N empfangenen E-Mails eines Apple-Mail-Kontos als .eml.
// Durchsucht alle (auch verschachtelten) Postfächer, überspringt Papierkorb,
// Gesendet, Entwürfe, Junk und Exchange-Sonderordner samt deren Unterordnern
// und entfernt Duplikate anhand der Message-ID. Nachrichten, die (auch) in einem
// ausgeschlossenen Ordner liegen, werden überall ausgeschlossen – nötig für Gmail,
// wo "Alle Nachrichten" auch Gesendetes und Entwürfe enthält.
//
// Aufruf:
//   osascript -l JavaScript scripts/export_mail.js                       # Konten auflisten
//   osascript -l JavaScript scripts/export_mail.js "<Konto>" 1000 data/raw ["Ordner1,Ordner2"]
//   osascript -l JavaScript scripts/export_mail.js --selftest            # ohne Mail
//
// Das optionale vierte Argument schließt weitere Ordner (samt Unterordnern) aus.

ObjC.import('Foundation');

const EXCLUDED_MAILBOXES = [
  'papierkorb', 'gelöschte elemente', 'deleted items', 'deleted messages', 'trash',
  'gesendet', 'gesendete elemente', 'gesendete objekte', 'gesendete nachrichten',
  'sent', 'sent items', 'sent messages', 'sent mail',
  'entwürfe', 'drafts',
  'junk', 'junk-e-mail', 'junk e-mail', 'junk email', 'spam',
  'postausgang', 'outbox',
  // Exchange-Sonderordner ohne normale E-Mails
  'synchronisierungsprobleme', 'sync issues', 'konflikte', 'conflicts',
  'lokale fehler', 'local failures', 'serverfehler', 'server failures',
  'aufgaben', 'tasks', 'notizen', 'notes', 'journal', 'kontakte', 'contacts',
  'kalender', 'calendar', 'erneut erinnern aktiviert', 'verlauf der unterhaltung',
  'conversation history',
];

const MAX_MAILBOX_DEPTH = 32;

// NFC, weil Mail Umlaute zerlegt liefert ("o" + U+0308) statt als "ö".
function normalize(name) {
  return String(name).normalize('NFC').trim().toLowerCase();
}

// Ein Postfach ist ausgeschlossen, wenn es selbst oder ein übergeordnetes
// Postfach auf der Ausschlussliste steht. pathNames: eigener Name + Vorfahren.
function isExcludedPath(pathNames, excluded) {
  return pathNames.some((n) => excluded.has(normalize(n)));
}

// Namen des Postfachs und aller übergeordneten Postfächer. Die Kette endet,
// wenn container() fehlschlägt (beim Konto selbst) oder nichts liefert.
function mailboxPath(mb) {
  const names = [mb.name()];
  let current = mb;
  for (let i = 0; i < MAX_MAILBOX_DEPTH; i++) {
    let parentName;
    try {
      current = current.container();
      parentName = current ? current.name() : null;
    } catch {
      break;
    }
    if (!parentName) break;
    names.push(parentName);
  }
  return names;
}

function compareDateDesc(a, b) {
  if (a.date && b.date) return b.date.getTime() - a.date.getTime();
  if (a.date) return -1;
  if (b.date) return 1;
  return 0;
}

// Wählt aus Kandidaten {key, messageId, date} die neuesten, ohne doppelte und ohne
// gesperrte Message-IDs. Kandidaten ohne Datum landen am Ende; ohne Message-ID
// wird weder dedupliziert noch gesperrt.
function selectLatest(candidates, blockedIds) {
  const sorted = candidates.slice().sort(compareDateDesc);
  const seen = new Set();
  const result = [];
  for (const c of sorted) {
    if (c.messageId) {
      if (seen.has(c.messageId) || blockedIds.has(c.messageId)) continue;
      seen.add(c.messageId);
    }
    result.push(c);
  }
  return result;
}

function pad(n, width) {
  return String(n).padStart(width, '0');
}

function isoDate(d) {
  if (!d) return 'unbekannt';
  return `${d.getFullYear()}-${pad(d.getMonth() + 1, 2)}-${pad(d.getDate(), 2)}`;
}

function resolvePath(p) {
  let s = $(p).stringByExpandingTildeInPath.js;
  if (!s.startsWith('/')) {
    s = `${$.NSFileManager.defaultManager.currentDirectoryPath.js}/${s}`;
  }
  return $(s).stringByStandardizingPath.js;
}

function ensureEmptyOutputDir(dir) {
  const fm = $.NSFileManager.defaultManager;
  if (!fm.createDirectoryAtPathWithIntermediateDirectoriesAttributesError(dir, true, $(), null)) {
    throw new Error(`Kann Ordner nicht anlegen: ${dir}`);
  }
  const entries = ObjC.deepUnwrap(fm.contentsOfDirectoryAtPathError(dir, null)) || [];
  const existing = entries.filter((f) => f.toLowerCase().endsWith('.eml'));
  if (existing.length > 0) {
    throw new Error(`${dir} enthält bereits ${existing.length} .eml-Dateien. Bitte vorher leeren.`);
  }
}

function writeFile(path, text) {
  const ok = $(text).writeToFileAtomicallyEncodingError(path, true, $.NSUTF8StringEncoding, null);
  if (!ok) throw new Error(`Schreiben fehlgeschlagen: ${path}`);
}

// account.mailboxes() liefert bei Mail bereits alle Postfächer einschließlich
// verschachtelter als flache Liste; daher keine Rekursion.
// Liefert {candidates, blockedIds}: Kandidaten aus erlaubten Postfächern und die
// Message-IDs aller Nachrichten aus ausgeschlossenen Postfächern.
function collectCandidates(account, excluded) {
  const candidates = [];
  const blockedIds = new Set();
  for (const mb of account.mailboxes()) {
    let path;
    try {
      path = mailboxPath(mb);
    } catch (e) {
      console.log(`  übersprungen (Name nicht lesbar): ${e.message}`);
      continue;
    }
    const label = path.slice(0, -1).reverse().join('/'); // ohne Kontoname
    if (isExcludedPath(path, excluded)) {
      try {
        for (const id of mb.messages.messageId()) if (id) blockedIds.add(id);
      } catch {
        // Sonderordner ohne Nachrichten (z. B. Notizen) – nichts zu sperren
      }
      continue;
    }
    try {
      const msgs = mb.messages;
      const ids = msgs.id();
      if (ids.length === 0) continue;
      // Sammelabfragen: eine Apple-Event-Anfrage pro Eigenschaft statt pro Nachricht.
      const dates = msgs.dateReceived();
      const messageIds = msgs.messageId();
      if (dates.length !== ids.length || messageIds.length !== ids.length) {
        throw new Error('Anzahl der Nachrichten hat sich beim Lesen geändert');
      }
      for (let i = 0; i < ids.length; i++) {
        candidates.push({ mailbox: mb, id: ids[i], messageId: messageIds[i], date: dates[i] });
      }
      console.log(`  ${label}: ${ids.length}`);
    } catch (e) {
      console.log(`  übersprungen ${label}: ${e.message}`);
    }
  }
  return { candidates, blockedIds };
}

function readSource(candidate) {
  try {
    return candidate.mailbox.messages.byId(candidate.id).source();
  } catch {
    return null; // Nachricht inzwischen verschoben/gelöscht
  }
}

function exportLatest(accountName, count, outDir, extraExcluded) {
  const Mail = Application('Mail');
  const matches = Mail.accounts().filter((a) => a.name() === accountName);
  if (matches.length !== 1) {
    throw new Error(`Konto "${accountName}" nicht gefunden. Ohne Argumente aufrufen, um Konten aufzulisten.`);
  }
  ensureEmptyOutputDir(outDir);

  const excluded = new Set([...EXCLUDED_MAILBOXES, ...extraExcluded].map(normalize));

  console.log('Lese Postfächer …');
  const { candidates, blockedIds } = collectCandidates(matches[0], excluded);
  const latest = selectLatest(candidates, blockedIds);
  console.log(`${latest.length} eindeutige Nachrichten gefunden, exportiere die neuesten ${count} …`);

  let written = 0;
  let skipped = 0;
  for (const c of latest) {
    if (written >= count) break;
    const source = readSource(c);
    if (!source) {
      skipped++; // Inhalt nicht lokal verfügbar oder Nachricht verschwunden
      continue;
    }
    written++;
    writeFile(`${outDir}/${pad(written, 4)}_${isoDate(c.date)}.eml`, source);
    if (written % 100 === 0) console.log(`  ${written}/${count}`);
  }
  return `Fertig: ${written} E-Mails nach ${outDir} exportiert, ${skipped} ohne Inhalt übersprungen.`;
}

function listAccounts() {
  const names = Application('Mail').accounts().map((a) => a.name());
  return `Verfügbare Konten:\n${names.map((n) => `  - ${n}`).join('\n')}`;
}

function assert(cond, msg) {
  if (!cond) throw new Error(`Selbsttest fehlgeschlagen: ${msg}`);
}

function selfTest() {
  const excluded = new Set([...EXCLUDED_MAILBOXES, 'Geheim'].map(normalize));
  const ex = (...path) => isExcludedPath(path, excluded);
  assert(ex('Papierkorb', 'Exchange') && ex(' Gesendete Elemente ', 'Exchange') && ex('Notizen', 'Exchange'),
    'Ausschlussliste');
  assert(ex('Datenschutz', 'Gelöschte Elemente', 'Exchange'), 'Unterordner ausgeschlossener Ordner');
  assert(ex('Gelöschte Elemente'.normalize('NFD')) && ex('Entwürfe'.normalize('NFD')),
    'zerlegte Umlaute (NFD), wie Mail sie liefert');
  assert(ex('Konflikte', 'Synchronisierungsprobleme', 'Exchange'), 'Exchange-Sonderordner');
  assert(ex('Unterordner', 'geheim', 'Exchange'), 'zusätzliche Ausschlüsse, Groß/klein egal');
  assert(!ex('Posteingang', 'Exchange') && !ex('KI', 'Areas', 'Posteingang', 'Exchange'),
    'normale Ordner nicht ausschließen');

  const d = (s) => new Date(s);
  const picked = selectLatest([
    { key: 'alt', messageId: '<a>', date: d('2026-01-01') },
    { key: 'neu', messageId: '<b>', date: d('2026-09-30') },
    { key: 'dup', messageId: '<b>', date: d('2026-09-30') },
    { key: 'ohneDatum', messageId: '<c>', date: null },
    { key: 'ohneDatum2', messageId: '<d>', date: null },
    { key: 'ohneId1', messageId: '', date: d('2026-05-01') },
    { key: 'ohneId2', messageId: '', date: d('2026-05-01') },
    { key: 'gesendet', messageId: '<s>', date: d('2026-10-01') },
  ], new Set(['<s>', ''])).map((c) => c.key);
  assert(JSON.stringify(picked) === JSON.stringify(['neu', 'ohneId1', 'ohneId2', 'alt', 'ohneDatum', 'ohneDatum2']),
    `Sortierung/Deduplizierung/Sperre: ${picked}`);

  assert(isoDate(new Date(2026, 8, 5)) === '2026-09-05', 'isoDate');
  assert(isoDate(null) === 'unbekannt', 'isoDate ohne Datum');

  const tmp = `${$.NSTemporaryDirectory().js}export_mail_selftest_${Date.now()}`;
  ensureEmptyOutputDir(tmp);
  writeFile(`${tmp}/0001_test.eml`, 'Subject: Grüße\n\nÄÖÜ ß');
  const back = $.NSString.stringWithContentsOfFileEncodingError(`${tmp}/0001_test.eml`, $.NSUTF8StringEncoding, null).js;
  assert(back === 'Subject: Grüße\n\nÄÖÜ ß', 'Umlaute beim Schreiben');
  let refused = false;
  try { ensureEmptyOutputDir(tmp); } catch { refused = true; }
  assert(refused, 'nicht-leeren Ordner ablehnen');
  $.NSFileManager.defaultManager.removeItemAtPathError(tmp, null);

  return 'Selbsttest OK';
}

function run(argv) {
  if (argv.length === 0) return listAccounts();
  if (argv[0] === '--selftest') return selfTest();
  if (argv.length !== 3 && argv.length !== 4) {
    return 'Aufruf: osascript -l JavaScript scripts/export_mail.js "<Konto>" <Anzahl> <Zielordner> ["Ordner1,Ordner2"]';
  }
  const count = parseInt(argv[1], 10);
  if (!(count > 0)) return `Ungültige Anzahl: ${argv[1]}`;
  const extraExcluded = argv.length === 4 ? argv[3].split(',').filter((s) => s.trim()) : [];
  return exportLatest(argv[0], count, resolvePath(argv[2]), extraExcluded);
}
