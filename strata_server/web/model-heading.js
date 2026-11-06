/** Gets the monitor heading and the paths behind the selected model.
 * @param {{name?: string, engine?: string | null, config_path?: string | null,
 *   model_path?: string | null} | null | undefined} details
 * @param {string} fallback */
export function getModelHeading(details, fallback) {
  const lines = [["Engine", details?.engine], ["Configuration", details?.config_path],
    ["Model", details?.model_path]];
  return {name: details?.name || fallback,
    title: lines.filter(([, value]) => value).map(([label, value]) => `${label}: ${value}`).join("\n")};
}
