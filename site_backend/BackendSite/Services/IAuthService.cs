using BackendSite.Models;

namespace BackendSite.Services
{
    public interface IAuthService
    {
        Task<(User? user, string? error)> AuthenticateAsync(string login, string password);
        Task<string> CreateSessionAsync(User user);
        Task<bool> ValidateSessionAsync(string sessionToken);
        Task<User?> GetUserFromSessionAsync(string sessionToken);
        Task LogoutAsync(string sessionToken);
    }
}