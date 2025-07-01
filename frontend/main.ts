import { app, BrowserWindow } from "electron";

const createWindow = async () => {
  const window = new BrowserWindow({
    width: 800,
    height: 600,
    title: "HouseCFG",
  });

  await window.loadFile("./dist/index.html");

  window.webContents.openDevTools();
};

app.whenReady().then(async () => {
  await createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
