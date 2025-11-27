using BackendSite.Services;

namespace BackendSite.Middleware
{
    public class SessionMiddleware
    {
        private readonly RequestDelegate _next;

        public SessionMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context, IAuthService authService, ICookieService cookieService)
        {
            var sessionToken = cookieService.GetSessionToken();

            if (!string.IsNullOrEmpty(sessionToken))
            {
                var user = await authService.GetUserFromSessionAsync(sessionToken);
                if (user != null)
                {
                    context.Items["UserId"] = user.Id;
                    context.Items["UserLogin"] = user.Login;
                }
            }

            await _next(context);
        }
    }
}