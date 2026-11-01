import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
// The server's web page draws the same panel from these two files (strata_server/web/app.js); vite.config.ts lets
// the dev server read them.
import { drawMemoryPanel, type ChangeSettings, type MemoryView } from '../../../../strata_server/web/memory-panel.js';
import '../../../../strata_server/web/memory-panel.css';

/** How often the panel reads the engine's memory again, in milliseconds (the web page: every 3 s too). */
const READ_EVERY_MS = 3000;

/** Shows what a serving model's engine holds in each memory (the web page's Memory panel), read through the host from
 *  its server's GET /v1/memory while the panel is on screen; nothing for an engine that does not report it. The
 *  panel's changes go to the server's POST /v1/memory through the host. The Monitor screen shows it for the chosen
 *  model. */
export function MemoryPanel({ instanceId }: { instanceId: string }) {
    const [view, setView] = useState<MemoryView | null>(null), element = useRef<HTMLDivElement>(null);
    const path = `/api/native/memory?instance=${encodeURIComponent(instanceId)}`;
    const change: ChangeSettings = (set, save) => api.post<MemoryView>(path, { set, save })
        .then(answer => ({ ok: true, view: answer }),
              (e: unknown) => ({ ok: false, error: e instanceof Error ? e.message : String(e) }));
    useEffect(() => {
        let live = true, timer: ReturnType<typeof setTimeout> | undefined;
        const read = async () => {
            if (!document.hidden) {
                const answer = await api.get<MemoryView>(path).catch(() => null);
                if (live)
                    setView(answer);
            }
            if (live)
                timer = setTimeout(read, READ_EVERY_MS);
        };
        void read();
        return () => { live = false; clearTimeout(timer); };
    }, [instanceId]);
    useEffect(() => {
        if (view?.available && element.current)
            drawMemoryPanel(element.current, view, change);
    }, [view]);
    return view?.available
        ? <section className="st-card monitor-memory">
            <h2 className="h4">{tr('Memory')}</h2>
            {/* a new element per instance: the panel keeps a drag, a message and an Undo per element */}
            <div ref={element} key={instanceId}/>
        </section>
        : null;
}
