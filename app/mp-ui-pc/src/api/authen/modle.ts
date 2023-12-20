export interface UserType {
    username: string
    password: string
    role: string
    roleId: string
    permissions: string | string[]
  }
  
export interface UserLogin {
  userName: string
  password: string
}