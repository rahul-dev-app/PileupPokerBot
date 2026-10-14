// Loads the extension's networks (extension/nets/*.bin) in Node.
const fs = require("fs");
const path = require("path");
const S = require("./extension/search.js");

function loadNets() {
  S.init(fs.readFileSync(path.join(__dirname, "extension", "mlp.wasm")));
  const dir = path.join(__dirname, "extension", "nets");
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "nets.json")));
  const load = (m) => { const b = fs.readFileSync(path.join(dir, m.file)); return S.loadNet(b.buffer.slice(b.byteOffset, b.byteOffset + b.length), m); };
  return S.makeNets(load(meta.value), load(meta.inner));
}

module.exports = { loadNets };
