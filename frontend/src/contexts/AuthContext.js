import React, { createContext, useContext, useState, useEffect } from 'react'
import axios from '../api/axios'
import { useNavigate } from 'react-router-dom'

const AuthContext = createContext()

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const loadUser = async () => {
    const token = localStorage.getItem('token')
    if (token) {
      try {
        const res = await axios.get('/users/me', {
          headers: { Authorization: `Bearer ${token}` },
        })
        setUser(res.data)
      } catch (err) {
        console.error('Invalid token')
        localStorage.removeItem('token')
        setUser(null)
      }
    }
    setLoading(false)
  }

  useEffect(() => {
    loadUser()
  }, [])

  const refreshUser = () => {
    loadUser()
  }

  const login = (token, userData) => {
    localStorage.setItem('token', token)
    setUser(userData)
    navigate('/dashboard')
  }

  const logout = () => {
    localStorage.removeItem('token')
    setUser(null)
    navigate('/login')
  }

  return (
    <AuthContext.Provider
      value={{ user, loading, login, logout, refreshUser, isAuthenticated: !!user }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  return useContext(AuthContext)
}
