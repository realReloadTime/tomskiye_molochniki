using BackendSite.Data;
using BackendSite.Models;
using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;

namespace BackendSite.Services
{
    public class AuthService : IAuthService
    {
        private readonly ReviewsHackatonDBContext _context;
        private readonly IPasswordHasherService _passwordHasher;

        public AuthService(ReviewsHackatonDBContext context, IPasswordHasherService passwordHasher)
        {
            _context = context;
            _passwordHasher = passwordHasher;
        }

        public async Task<(User? user, string? error)> AuthenticateAsync(string login, string password)
        {
            var user = await _context.Users
                .FirstOrDefaultAsync(u => u.Login == login);

            if (user == null)
                return (null, "Пользователя с таким Логином не существует");

            if (!_passwordHasher.VerifyPassword(password, user.PasswordHash))
                return (null, "Неверный логин или пароль");

            return (user, null);
        }

        public async Task<string> CreateSessionAsync(User user)
        {
            var sessionToken = GenerateSessionToken();
            user.SessionToken = sessionToken;
            user.SessionExpires = DateTime.UtcNow.AddDays(30);
            await _context.SaveChangesAsync();
            return sessionToken;
        }

        public async Task<bool> ValidateSessionAsync(string sessionToken)
        {
            if (string.IsNullOrEmpty(sessionToken))
                return false;

            var user = await _context.Users
                .FirstOrDefaultAsync(u =>
                    u.SessionToken == sessionToken &&
                    u.SessionExpires > DateTime.UtcNow);

            return user != null;
        }

        public async Task<User?> GetUserFromSessionAsync(string sessionToken)
        {
            if (string.IsNullOrEmpty(sessionToken))
                return null;

            return await _context.Users
                .FirstOrDefaultAsync(u =>
                    u.SessionToken == sessionToken &&
                    u.SessionExpires > DateTime.UtcNow);
        }

        public async Task LogoutAsync(string sessionToken)
        {
            var user = await _context.Users
                .FirstOrDefaultAsync(u => u.SessionToken == sessionToken);

            if (user != null)
            {
                user.SessionToken = null;
                user.SessionExpires = null;
                await _context.SaveChangesAsync();
            }
        }

        private string GenerateSessionToken()
        {
            var randomBytes = new byte[32];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(randomBytes);
            return Convert.ToBase64String(randomBytes);
        }
    }
}