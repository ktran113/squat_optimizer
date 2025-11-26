CREATE TABLE users{
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE,
    name TEXT,
    created_at TIMESTAMP DEFAULT now()
};