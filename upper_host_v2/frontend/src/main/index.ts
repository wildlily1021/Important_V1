import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import { statSync } from 'fs'
import { join } from 'path'

import icon from '../../resources/icon.png?asset'

const isDev = !app.isPackaged

app.disableHardwareAcceleration()
app.commandLine.appendSwitch('disable-gpu')
app.commandLine.appendSwitch('disable-gpu-compositing')
app.commandLine.appendSwitch('disable-gpu-rasterization')
app.commandLine.appendSwitch('disable-accelerated-2d-canvas')
app.commandLine.appendSwitch('disable-gpu-sandbox')
app.commandLine.appendSwitch('use-angle', 'swiftshader')
app.commandLine.appendSwitch('use-gl', 'swiftshader')
app.commandLine.appendSwitch('enable-unsafe-swiftshader')
app.commandLine.appendSwitch('no-proxy-server')
app.commandLine.appendSwitch('proxy-bypass-list', '<-loopback>')

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1680,
    height: 1020,
    minWidth: 1280,
    minHeight: 820,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#2f2f33',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (isDev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  if (process.platform === 'win32') {
    app.setAppUserModelId(isDev ? process.execPath : 'com.electron')
  }

  ipcMain.handle('signal:select-file', async () => {
    const result = await dialog.showOpenDialog({
      title: '选择信号文件',
      properties: ['openFile'],
      filters: [
        { name: 'Signal data', extensions: ['txt', 'csv', 'xlsx', 'xls'] },
        { name: 'All files', extensions: ['*'] }
      ]
    })

    if (result.canceled || result.filePaths.length === 0) {
      return null
    }

    const filePath = result.filePaths[0]
    const stat = statSync(filePath)

    return {
      path: filePath,
      name: filePath.split(/[\\/]/).pop() ?? filePath,
      size: stat.size
    }
  })

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
