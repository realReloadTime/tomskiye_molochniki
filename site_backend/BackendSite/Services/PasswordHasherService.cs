using Microsoft.AspNetCore.Identity;
using BackendSite.Models;

namespace BackendSite.Services
{
    public interface IPasswordHasherService
    {
        string HashPassword(string password);
        bool VerifyPassword(string password, string hashedPassword);
    }

    public class PasswordHasherService : IPasswordHasherService
    {
        private readonly PasswordHasher<User> _passwordHasher;

        public PasswordHasherService()
        {
            _passwordHasher = new PasswordHasher<User>();
        }

        public string HashPassword(string password)
        {
            return _passwordHasher.HashPassword(null, password);
        }

        public bool VerifyPassword(string password, string hashedPassword)
        {
            var result = _passwordHasher.VerifyHashedPassword(null, hashedPassword, password);
            return result == PasswordVerificationResult.Success;
        }
    }
}