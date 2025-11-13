// Mock authentication using localStorage

export interface User {
  id: string;
  email: string;
  name: string;
}

const AUTH_KEY = 'boq_auth_user';
const TOKEN_KEY = 'boq_auth_token';

export const login = async (email: string, password: string): Promise<{ user: User; token: string } | null> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  // Simple validation
  if (!email || !password) {
    throw new Error('Email and password are required');
  }
  
  if (password.length < 6) {
    throw new Error('Invalid credentials');
  }

  const user: User = {
    id: `user_${Date.now()}`,
    email,
    name: email.split('@')[0],
  };

  const token = `mock_token_${Date.now()}`;
  
  localStorage.setItem(AUTH_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, token);
  
  return { user, token };
};

export const signup = async (email: string, password: string, name: string): Promise<{ user: User; token: string }> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  if (!email || !password || !name) {
    throw new Error('All fields are required');
  }
  
  if (password.length < 6) {
    throw new Error('Password must be at least 6 characters');
  }

  const user: User = {
    id: `user_${Date.now()}`,
    email,
    name,
  };

  const token = `mock_token_${Date.now()}`;
  
  localStorage.setItem(AUTH_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, token);
  
  return { user, token };
};

export const logout = () => {
  localStorage.removeItem(AUTH_KEY);
  localStorage.removeItem(TOKEN_KEY);
};

export const getCurrentUser = (): User | null => {
  const userStr = localStorage.getItem(AUTH_KEY);
  if (!userStr) return null;
  try {
    return JSON.parse(userStr);
  } catch {
    return null;
  }
};

export const getToken = (): string | null => {
  return localStorage.getItem(TOKEN_KEY);
};

export const isAuthenticated = (): boolean => {
  return !!getToken() && !!getCurrentUser();
};
