import assert = require("node:assert");

import { contextBridge } from "electron";
import { electronAPI } from "@electron-toolkit/preload";

import apiBase from "@/config/api.ts";

// Custom APIs for renderer
const api = {};

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld("electron", electronAPI);
    contextBridge.exposeInMainWorld("api", api);
  } catch (error) {
    console.error(error);
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI;
  // @ts-ignore (define in dts)
  window.api = api;
}

async function isServerRunning() {
  try {
    const response = await fetch(`${apiBase}/ping`);

    const data = await response.json();

    return data?.message === "pong";
  } catch (error) {
    console.error("Server check failed:", error);

    return false;
  }
}

async function serverCheck() {
  if (await isServerRunning()) {
    console.log("Server is running");

    return;
  }
}
