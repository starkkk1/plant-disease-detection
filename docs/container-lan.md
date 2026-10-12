# Phone access to the local container

The Docker image `plant-disease-api:v2` contains the FastAPI service on port 8000. It does not serve the Next.js website on port 3000; start that frontend separately.

## Current addresses

The Wi-Fi IPv4 address observed on 2026-10-10 is **192.168.2.48**. Open **http://192.168.2.48:3000** on a phone connected to the same local network. The direct API readiness URL is **http://192.168.2.48:8000/health**. Use the Wi-Fi address, not the Docker/WSL virtual address or `localhost` on the phone. If DHCP changes the computer's IP, update these URLs and CORS.

## Start the services

The configured container is `plant-disease-api-lan`. If it already exists, use `docker start plant-disease-api-lan`; do not create a second container on the same port. For a new installation with the existing image:

```powershell
docker run -d --name plant-disease-api-lan --restart unless-stopped -p 0.0.0.0:8000:8000 -e "CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://192.168.2.48:3000" plant-disease-api:v2
```

Docker's [port-publishing documentation](https://docs.docker.com/engine/network/port-publishing/) explains why binding a published port to `127.0.0.1` restricts access to the host. The configured API uses an all-interface host binding and an internal Uvicorn `0.0.0.0` binding.

Run the frontend from another terminal:

```powershell
Set-Location D:\Code\python\plant-disease-detection\frontend
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Keep `NEXT_PUBLIC_API_URL` unset for the LAN demo: the frontend derives the phone page's hostname and port 8000. If explicitly configured, it must point to an address reachable from the phone, not `http://localhost:8000`. Changing a public frontend environment variable requires restarting the dev server or rebuilding a production image.

## Firewall and network checks

The observed Wi-Fi profile is **Private**. Existing Node/Docker application rules were enabled for **Public**, so they did not authorize inbound traffic on this Wi-Fi profile. After explicit user approval on 2026-10-10, the enabled rule **`PlantDisease-LAN-Private`** was created and verified: inbound Allow, TCP ports 3000/8000, local address 192.168.2.48, profile Private, and remote `LocalSubnet`. Windows required Administrator access; the operation completed through UAC. The initial automatic-review rejection made no firewall change; the rule was added only after approval.

This rule persists across restarts. To remove it later, run the following in an Administrator PowerShell:

```powershell
Remove-NetFirewallRule -Name PlantDisease-LAN-Private
```

If the page cannot open, first try `/health` on the phone. If neither address opens while both work on the computer, check the approved firewall rule and whether the phone is on guest Wi-Fi, a VPN, or a network with client isolation. If the page opens but prediction fails, check the API origin and the CORS origin. Do not disable the whole firewall or use wildcard CORS to solve the LAN problem.

A browser test performed on the computer using its own LAN address does not prove that a physical phone can reach it. Runtime evidence is recorded in `reports/sprint-01/lan-container-check.json`, including the actual container model version and the separate physical-phone/firewall statuses.

## Logs and stopping

Use `docker logs --tail 50 plant-disease-api-lan` for API logs. The frontend started during diagnosis writes `.tmp/frontend-lan.stdout.log` and `.tmp/frontend-lan.stderr.log`. `docker stop plant-disease-api-lan` stops this API container. The container's restart policy retains it across Docker restarts unless explicitly stopped.
