// Settings popup: mode (compete / suggest) and the optional analytics endpoint, stored in chrome.storage.sync.
const DEFAULTS = { mode: "compete", analyticsUrl: "" };
const analytics = document.getElementById("analytics");
const saved = document.getElementById("saved");

function save(values, text = "Saved") {
  chrome.storage.sync.set(values, () => {
    saved.textContent = text;
    setTimeout(() => (saved.textContent = ""), 1500);
  });
}

chrome.storage.sync.get(DEFAULTS, (s) => {
  document.querySelector(`input[value="${s.mode}"]`).checked = true;
  analytics.value = s.analyticsUrl;
});
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => save({ mode: r.value })));

// A custom endpoint needs the permission to send to its origin (asked once, on Save).
document.getElementById("save").addEventListener("click", async () => {
  const url = analytics.value.trim();
  if (!url) return save({ analyticsUrl: "" }, "Saved: no analytics");
  let origin;
  try {
    origin = new URL(url).origin;
  } catch {
    saved.textContent = "Not a valid URL";
    return;
  }
  const granted = await chrome.permissions.request({ origins: [origin + "/*"] });
  if (granted) save({ analyticsUrl: url });
  else saved.textContent = "Permission refused: not saved";
});
