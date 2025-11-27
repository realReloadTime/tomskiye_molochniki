using Microsoft.AspNetCore.Identity;
using BackendSite.Models;

namespace BackendSite.Services
{
    public class PasswordHasherService : IPasswordHasherService
    {
        private readonly PasswordHasher<User> _passwordHasher;

        public PasswordHasherService()
        {
            _passwordHasher = new PasswordHasher<User>();
        }

        public string HashPassword(string password)
        {

            var tempUser = new User();
            return _passwordHasher.HashPassword(tempUser, password);
        }

        public bool VerifyPassword(string password, string hashedPassword)
        {
            var tempUser = new User();
            var result = _passwordHasher.VerifyHashedPassword(tempUser, hashedPassword, password);
            return result == PasswordVerificationResult.Success;
        }
    }
}