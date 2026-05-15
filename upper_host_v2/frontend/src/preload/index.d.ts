import { ElectronAPI } from '@electron-toolkit/preload'

export type SelectedSignalFile = {
  path: string
  name: string
  size: number
}

export type AppAPI = {
  selectSignalFile: () => Promise<SelectedSignalFile | null>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: AppAPI
  }
}
